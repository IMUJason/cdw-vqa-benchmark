"""QLoRA SFT for Qwen3-VL-8B on CDW-VQA instruction data.
Custom loop (no external trainer): completion-only loss, LoRA on LLM-side
linears, 4-bit base, bf16 compute. Usage: python scripts/train_qlora.py [--smoke]"""
import sys, json, math, os, time, argparse
from pathlib import Path
import random

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
DATA = BASE / "data"
CACHE = Path(r"datasets/plan3_sft_cache")
OUT_DIR = Path(os.environ.get("OUT_DIR", r"models/adapters/qwen3vl-cdw-lora-r16"))

MODEL_PATH = os.environ.get("MODEL_PATH", r"models/qwen3-vl-8b-instruct")
MAX_SIDE = 768
EPOCHS = 2
LR = 1e-4
ACCUM = 8
LORA_R, LORA_ALPHA, LORA_DROP = 16, 32, 0.05


def resized(path: str) -> str:
    import cv2, numpy as np
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / (Path(path).stem + ".jpg")
    if out.exists():
        return str(out)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    buf.tofile(str(out))
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import random as _r
    _seed = int(os.environ.get('SEED', '0'))
    if _seed:
        _r.seed(_seed)
        import torch as _t
        _t.manual_seed(_seed)
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig, get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    lconf = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROP,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"],
                       task_type="CAUSAL_LM")
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    items = [json.loads(l) for l in open(
        os.environ.get("SFT_FILE", str(DATA / "sft_train.jsonl")), encoding="utf-8")]
    if args.smoke:
        items = items[:48]

    class SFTData(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            img_path = resized(it["image"])
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": it["question"]}]}]
            prompt = processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            imgs, _ = process_vision_info(msgs)
            full = prompt + it["answer"] + "<|im_end|>\n"
            p_in = processor(text=[prompt], images=imgs, return_tensors="pt")
            f_in = processor(text=[full], images=imgs, return_tensors="pt")
            n_p = p_in["input_ids"].shape[1]
            ids = f_in["input_ids"][0]
            labels = ids.clone()
            labels[:n_p] = -100
            return dict(input_ids=ids, labels=labels,
                        mm_ids=f_in.get("mm_token_type_ids",
                                        torch.zeros_like(f_in["input_ids"]))[0],
                        pixel_values=f_in.get("pixel_values"),
                        grid=f_in.get("image_grid_thw"))

    def collate(batch):
        pad_id = processor.tokenizer.pad_token_id or 151643
        maxlen = max(b["input_ids"].shape[0] for b in batch)
        ids, labs, attn, mm = [], [], [], []
        for b in batch:
            n = maxlen - b["input_ids"].shape[0]
            ids.append(torch.cat([b["input_ids"],
                                  torch.full((n,), pad_id, dtype=torch.long)]))
            labs.append(torch.cat([b["labels"],
                                   torch.full((n,), -100, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones_like(b["input_ids"]),
                                   torch.zeros(n, dtype=torch.long)]))
            mm.append(torch.cat([b["mm_ids"], torch.zeros(n, dtype=torch.long)]))
        out = dict(input_ids=torch.stack(ids), labels=torch.stack(labs),
                   attention_mask=torch.stack(attn),
                   mm_token_type_ids=torch.stack(mm))
        if batch[0]["pixel_values"] is not None:
            out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch])
            out["image_grid_thw"] = torch.cat([b["grid"] for b in batch])
        return out

    dl = DataLoader(SFTData(items), batch_size=1, shuffle=True,
                    collate_fn=collate, num_workers=0)
    steps_total = len(dl) * EPOCHS // ACCUM
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps_total), steps_total)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gstep = 0
    running = 0.0
    for ep in range(EPOCHS):
        for i, batch in enumerate(dl):
            batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(**batch)
            (out.loss / ACCUM).backward()
            running += out.loss.item()
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 20 == 0:
                    el = time.time() - t0
                    print(f"ep{ep} step {gstep}/{steps_total} "
                          f"loss {running / (20 * ACCUM):.4f} "
                          f"{el / gstep:.1f}s/step", flush=True)
                    running = 0.0
            if args.smoke and gstep >= 5:
                break
        if args.smoke and gstep >= 5:
            break
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print("SAVED", OUT_DIR, "| steps:", gstep)


if __name__ == "__main__":
    main()
