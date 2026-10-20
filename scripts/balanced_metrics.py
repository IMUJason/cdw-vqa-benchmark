"""Clean balanced metrics using the OFFICIAL stored 'correct' field:
- object balanced accuracy = mean per-class recall (group stored correct by cls)
- material balanced accuracy on pipes only (PVC vs metal recall)
- majority baselines; top material confusions on pipes."""
import json
from pathlib import Path
from collections import Counter, defaultdict
import re

BASE = Path(__file__).resolve().parents[1]


def clean(a):
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
    a = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</?answer>", "", a)
    return a.strip().lower()


def obj_balanced(recs):
    per = defaultdict(lambda: [0, 0])
    for r in recs:
        per[r["cls"]][1] += 1
        per[r["cls"]][0] += r["correct"]
    return sum(c / t for c, t in per.values()) / len(per), dict(per)


def mat_balanced_pipes(recs):
    per = defaultdict(lambda: [0, 0])
    for r in recs:
        a = clean(r["raw"])
        pr = "metal" if "metal" in a else ("PVC" if ("pvc" in a or "plastic" in a) else "other")
        per[r["answer"]][1] += 1
        per[r["answer"]][0] += (pr == r["answer"])
    return sum(c / t for c, t in per.values()) / len(per), dict(per)


def load(fp):
    return [json.loads(l) for l in open(fp, encoding="utf-8")]


def report(name, recs):
    obj = [r for r in recs if r["family"] == "instance_object"]
    mat = [r for r in recs if r["family"] == "instance_material" and r["cls"] == "pipes"]
    ob, oper = obj_balanced(obj)
    mb, mper = mat_balanced_pipes(mat)
    recalls = {g: f"{c}/{t}={c/t:.2f}" for g, (c, t) in sorted(mper.items())}
    print(f"{name:14s} obj_bal={ob:.3f}  mat_bal(pipes)={mb:.3f}  PVC/metal recall: {recalls}")


print("== Majority baselines ==")
print("pipes material always-PVC = 0.885 | object always-pipes = 0.722\n")

print("== Balanced accuracy (official scorer) ==")
for mk, fp in [("zs_qwen3", BASE / "results/w6/responses_qwen3-vl-8b.jsonl"),
               ("zs_qwen25", BASE / "results/w6/responses_qwen2.5-vl-7b.jsonl"),
               ("zs_glm41", BASE / "results/w6/responses_glm-4.1v-9b.jsonl")]:
    report(mk, load(fp))
for mk in ["kimi", "glm"]:
    recs = [r for r in load(BASE / f"results/api_eval/api_{mk}.jsonl")
            if "diag_id" in r and not str(r["raw"]).startswith("ERROR")]
    report(f"api_{mk}", recs)
for mk, fp in [("ad_qwen3", BASE / "results/adapted/adapted_qwen3vl.jsonl"),
               ("ad_qwen25", BASE / "results/adapted/adapted_qwen25vl.jsonl")]:
    report(mk, [r for r in load(fp) if "diag_id" in r])
