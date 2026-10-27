"""QLoRA SFT for GLM-4.1V-9B-Thinking (GLM-specific data prep).
Same recipe as train_qlora.py but without qwen_vl_utils.
Usage: python scripts/train_qlora_glm.py [--smoke]"""
import sys, json, time, argparse, os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
DATA = BASE / "data"
CACHE = Path(r"datasets/plan3_sft_cache")
OUT_DIR = Path(os.environ.get("OUT_DIR", r"models/adapters/glm41v-cdw-lora-r16"))
MODEL_PATH = r"models/glm-4.1v-9b-thinking"
MAX_SIDE, EPOCHS, LR, ACCUM = 512, 2, 1e-4, 8
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

    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
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
    if os.environ.get("GRAD_CKPT", "1") != "0":
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    eos = processor.tokenizer.eos_token or "<|endoftext|>"

    items = [json.loads(l) for l in open(DATA / "sft_train.jsonl", encoding="utf-8")]
    if args.smoke:
        items = items[:48]

    class SFTData(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            img = Image.open(resized(it["image"])).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": resized(it["image"])},
                {"type": "text", "text": it["question"]}]}]
            prompt = processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            full = prompt + it["answer"] + eos
            # single processor call: encode full text+image once;
            # answer length via tokenizer-only (no image) for label masking
            f_in = processor(text=[full], images=[img], return_tensors="pt")
            ans_ids = processor.tokenizer(it["answer"] + eos,
                                          return_tensors="pt",
                                          add_special_tokens=False)
            n_ans = ans_ids["input_ids"].shape[1]
            ids = f_in["input_ids"][0]
            labels = ids.clone()
            labels[:ids.shape[0] - n_ans] = -100
            keys = dict(input_ids=ids, labels=labels,
                        mm_ids=f_in.get("mm_token_type_ids",
                                        torch.zeros_like(f_in["input_ids"]))[0])
            if f_in.get("pixel_values") is not None:
                keys["pixel_values"] = f_in["pixel_values"]
                keys["grid"] = f_in.get("image_grid_thw")
            return keys

    def collate(batch):
        pad_id = processor.tokenizer.pad_token_id or 0
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
        if "pixel_values" in batch[0]:
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
                if gstep % 100 == 0:
                    ck = OUT_DIR.parent / (OUT_DIR.name + f"-step{gstep}")
                    model.save_pretrained(ck)
                    print(f"CHECKPOINT {ck} | step {gstep}", flush=True)
                if gstep % 20 == 0:
                    print(f"ep{ep} step {gstep}/{steps_total} "
                          f"loss {running / (20 * ACCUM):.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
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
