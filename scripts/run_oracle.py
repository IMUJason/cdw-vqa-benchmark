"""Ask an adapted model on oracle-boxed L0 pipes (test split), score, compare.

Env overrides: ORACLE_BASE, ORACLE_ADAPTER, ORACLE_NAME (output filename stem).
Default = adapted Qwen3-VL. Uses the symmetric two-level scorer from
fine_metrics for object and material questions.
"""
import os, sys, json
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from fine_metrics import obj_score, mat_score

DATA = BASE / "data"
ARMS = Path(r"datasets/plan3_arms")
OUT = BASE / "results" / "oracle"
OUT.mkdir(exist_ok=True)
NAME = os.environ.get("ORACLE_NAME", "qwen3vl")

qa = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
qa = [it for it in qa if it["arm"] == "L0" and it["cls"] == "pipes"
      and it["split"] == "test" and not it.get("excluded") and not it.get("needs_manual_label")]
print("oracle items:", len(qa))

model = AutoModelForImageTextToText.from_pretrained(
    os.environ.get("ORACLE_BASE", r"models/qwen3-vl-8b-instruct"),
    dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(
    model, os.environ.get("ORACLE_ADAPTER", r"models/adapters/qwen3vl-cdw-lora-r16"))
model.eval()
processor = AutoProcessor.from_pretrained(
    os.environ.get("ORACLE_BASE", r"models/qwen3-vl-8b-instruct"))


def ask(q, img):
    msgs = [{"role": "user", "content": [{"type": "image", "image": str(img)},
                                         {"type": "text", "text": q}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    imgs, _ = process_vision_info(msgs)
    inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()


res = []
for i, it in enumerate(qa):
    img = ARMS / "oracle_box" / f"{it['diag_id']:03d}_L0box.jpg"
    q = it["question"] + " Answer with one word or a short phrase."
    raw = ask(q, img)
    res.append(dict(it, raw=raw))
    if i % 40 == 0:
        print(i, flush=True)

with open(OUT / f"oracle_boxed_{NAME}.jsonl", "w", encoding="utf-8") as f:
    for r in res:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\n==== oracle boxed-L0:", NAME, "====")
for fam in ["instance_object", "instance_material"]:
    sub = [r for r in res if r["family"] == fam]
    if fam == "instance_object":
        acc = sum(obj_score(r["raw"]) for r in sub) / len(sub)
        print(f"object  n={len(sub)} acc={acc:.3f}")
    else:
        fine = sum(mat_score(r["raw"], r["answer"])[0] for r in sub) / len(sub)
        coarse = sum(mat_score(r["raw"], r["answer"])[1] for r in sub) / len(sub)
        print(f"material n={len(sub)} fine={fine:.3f} coarse={coarse:.3f}")
