"""L0-only ablation SFT set: scene train + instance train restricted to arm L0.

Motivation: the original SFT set
includes all four arms, so the L1/L2/L3 test gains are confounded with having
seen the rendering transformations (mean-fill inpaint / blur / crop) during
training. This ablation trains on L0 instance views only and is evaluated on
L1-L3 to test transfer.
"""
import sys, json, random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
DATA = BASE / "data"
random.seed(20261012)

SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test",
             "train": "train", "val": "val", "test": "test"}


def fmt_q(it):
    q = it["question"]
    if it.get("qtype") == "mcq":
        ch = " ".join(f"{k}. {v}" for k, v in it["choices"].items())
        return f"{q}\nChoices: {ch}\nAnswer with the letter only.", it["answer"]
    if it.get("qtype") == "yesno":
        return f"{q}\nAnswer Yes or No.", it["answer"]
    return f"{q} Answer with one word or a short phrase.", it["answer"]


items = []
for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if SPLIT_MAP.get(it["split"]) != "train":
        continue
    q, a = fmt_q(it)
    items.append(dict(image=it["image_path"], question=q, answer=a,
                      source="scene", family=it["family"], dataset=it["dataset"]))
for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if it.get("excluded") or it.get("needs_manual_label"):
        continue
    if it["split"] != "train" or it["arm"] != "L0":
        continue
    q, a = fmt_q(it)
    items.append(dict(image=str(Path(r"datasets/plan3_arms") / it["image"]),
                      question=q, answer=a, source="instance",
                      family=it["family"], dataset="codd_diag"))

random.shuffle(items)
out = DATA / "sft_train_l0.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
from collections import Counter
print("L0-only sft items:", len(items))
print(Counter(f"{i['source']}_{i['family']}" for i in items))
