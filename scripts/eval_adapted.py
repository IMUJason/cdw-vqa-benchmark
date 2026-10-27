"""Adapted-model evaluation: base+LoRA adapter on TEST splits (scene+instance).
Baseline numbers come from existing zero-shot responses (scene_eval/w6).
Usage: python scripts/eval_adapted.py qwen3vl"""
import sys, os, json, re, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from run_scene_eval import prompt_of, score as scene_score, clean, cached_image
from run_w6_gate import score as inst_score, MODELS as W6_MODELS

DATA = BASE / "data"
OUT = BASE / "results" / "adapted"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = Path(r"datasets/plan3_arms")

CONFIGS = {
    "qwen3vl": dict(base=r"models/qwen3-vl-8b-instruct",
                    adapter=r"models/adapters/qwen3vl-cdw-lora-r16",
                    max_tokens=64),
    "qwen3vl_v2": dict(base=r"models/qwen3-vl-8b-instruct",
                       adapter=r"models/adapters/qwen3vl-cdw-lora-r16-v2",
                       max_tokens=64),
    "qwen25vl_v2": dict(base=r"models/qwen2.5-vl-7b-instruct",
                        adapter=r"models/adapters/qwen25vl-cdw-lora-r16-v2",
                        max_tokens=64),
    "glm41v_v2": dict(base=r"models/glm-4.1v-9b-thinking",
                      adapter=r"models/adapters/glm41v-cdw-lora-r16-v2",
                      max_tokens=1024),
    "qwen3vl_seed2": dict(base=r"models/qwen3-vl-8b-instruct",
                          adapter=r"models/adapters/qwen3vl-cdw-lora-r16-seed2",
                          max_tokens=64),
    "glm41v": dict(base=r"models/glm-4.1v-9b-thinking",
                   adapter=r"models/adapters/glm41v-cdw-lora-r16",
                   max_tokens=1024),
    "qwen25vl": dict(base=r"models/qwen2.5-vl-7b-instruct",
                     adapter=r"models/adapters/qwen25vl-cdw-lora-r16",
                     max_tokens=64),
}
SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test",
             "train": "train", "val": "val", "test": "test"}


def main():
    key = sys.argv[1]
    conf = CONFIGS[key]
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    model = AutoModelForImageTextToText.from_pretrained(
        conf["base"], dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, conf["adapter"])
    model.eval()
    processor = AutoProcessor.from_pretrained(conf["base"])

    def ask(q, img, max_tokens):
        content = [{"type": "image", "image": str(img)},
                   {"type": "text", "text": q}]
        msgs = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        imgs, _ = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()

    # scene test
    scene = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
    scene = [it for it in scene if SPLIT_MAP[it["split"]] == "test"]
    # instance test
    inst = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
    inst = [it for it in inst if not it.get("excluded") and not it.get("needs_manual_label")
            and it["split"] == "test"]
    print(f"scene test: {len(scene)}, instance test: {len(inst)}")

    resp_path = OUT / (os.environ.get("OUT_NAME", f"adapted_{key}") + ".jsonl")
    done = set()
    if resp_path.exists():
        for l in open(resp_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r.get("family"), r.get("image_id") or str(r.get("diag_id")),
                      r.get("arm", "scene"), r["question"][:40]))
    t0 = time.time()
    with open(resp_path, "a", encoding="utf-8") as f:
        n = 0
        for it in scene:
            k = (it["family"], it["image_id"], "scene", it["question"][:40])
            if k in done:
                continue
            raw = ask(prompt_of(it), cached_image(it["image_path"]), conf["max_tokens"])
            f.write(json.dumps(dict(it, raw=raw, correct=scene_score(it, raw)),
                               ensure_ascii=False) + "\n")
            f.flush(); n += 1
            if n % 200 == 0:
                print(f"scene {n} {time.time() - t0:.0f}s", flush=True)
        for it in inst:
            k = (it["family"], str(it["diag_id"]), it["arm"], it["question"][:40])
            if k in done:
                continue
            q = it["question"] + " Answer with one word or a short phrase."
            raw = ask(q, ARMS / it["image"], conf["max_tokens"])
            f.write(json.dumps(dict(it, raw=raw, correct=inst_score(it, raw)),
                               ensure_ascii=False) + "\n")
            f.flush(); n += 1
            if n % 200 == 0:
                print(f"inst {n} {time.time() - t0:.0f}s", flush=True)
    print("DONE", key, n)


if __name__ == "__main__":
    main()
