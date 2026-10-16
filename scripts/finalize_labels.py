"""Finalize pipes labels: 3-judge protocol.
auto_accept = initial agreement (Qwen==GLM positive) + Kimi-resolved disputes
(GLM positive AND Kimi same positive -> 2/3 majority positive).
to_judge = everything else. Audit 15% of auto_accept."""
import csv, random
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
random.seed(20261010)

q = {r["diag_id"]: r for r in csv.DictReader(open(DATA / "pipes_prefill.csv", encoding="utf-8-sig"))}
g = {r["diag_id"]: r for r in csv.DictReader(open(DATA / "pipes_prefill_glm.csv", encoding="utf-8-sig"))}
k = {r["diag_id"]: r for r in csv.DictReader(open(DATA / "pipes_prefill_kimi.csv", encoding="utf-8-sig"))}

accept, judge = [], []
for did, qr in q.items():
    ql, gl = qr["prefill(metal/PVC/other)"], g[did]["glm_prefill"]
    kr = k.get(did)
    kl = kr["kimi_prefill"] if kr else None
    kraw = kr["kimi_raw"] if kr else ""
    final = None
    if ql == gl and ql in ("metal", "PVC"):
        final = ql
    elif gl == kl and gl in ("metal", "PVC"):
        final = gl
    elif ql == kl and ql in ("metal", "PVC"):
        final = ql
    votes = f"{ql}/{gl}/{kl}"
    if final:
        accept.append([did, qr["image_L3"], final, votes,
                       qr["raw_answer"], g[did]["glm_raw"], kraw])
    else:
        judge.append([did, qr["image_L3"], "", votes,
                      qr["raw_answer"], g[did]["glm_raw"], kraw])

n_audit = max(1, int(0.15 * len(accept)))
for r in random.sample(accept, k=n_audit):
    r.append("AUDIT")

with open(DATA / "pipes_auto_accept.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["diag_id", "image_L3", "auto_label", "votes(q/g/k)",
                "qwen_raw", "glm_raw", "kimi_raw", "audit_flag"])
    w.writerows(accept)
with open(DATA / "pipes_to_judge.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["diag_id", "image_L3", "material_label(人工判别填这里)",
                "votes(qwen/glm/kimi)", "qwen_raw", "glm_raw", "kimi_raw"])
    w.writerows(judge)
print(f"final auto_accept: {len(accept)} (audit {n_audit})")
print(f"final to_judge:    {len(judge)}")
