"""Merge Qwen3+GLM pipes labels into auto_accept / to_judge (dual-judge protocol).
auto_accept: both models give the SAME positive label (metal/PVC), 15% audit flag.
to_judge: disagreement or any 'other' — human adjudication required."""
import csv, random
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
random.seed(20261010)

q = {r["diag_id"]: r for r in csv.DictReader(open(DATA / "pipes_prefill.csv", encoding="utf-8-sig"))}
g = {r["diag_id"]: r for r in csv.DictReader(open(DATA / "pipes_prefill_glm.csv", encoding="utf-8-sig"))}

accept, judge = [], []
for did, qr in q.items():
    qlab = qr["prefill(metal/PVC/other)"]
    glab = g[did]["glm_prefill"]
    if qlab == glab and qlab in ("metal", "PVC"):
        accept.append([did, qr["image_L3"], qlab, qr["raw_answer"], g[did]["glm_raw"]])
    else:
        judge.append([did, qr["image_L3"], "", qlab, glab, qr["raw_answer"], g[did]["glm_raw"]])

for r in random.sample(accept, k=max(1, int(0.15 * len(accept)))):
    r.append("AUDIT")

with open(DATA / "pipes_auto_accept.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["diag_id", "image_L3", "auto_label", "qwen_raw", "glm_raw", "audit_flag"])
    w.writerows(accept)
with open(DATA / "pipes_to_judge.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["diag_id", "image_L3", "material_label(人工判别填这里)",
                "qwen_prefill", "glm_prefill", "qwen_raw", "glm_raw"])
    w.writerows(judge)
print(f"auto_accept: {len(accept)} (audit {sum(1 for r in accept if len(r) > 5)})")
print(f"to_judge:    {len(judge)}")
