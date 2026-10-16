"""Scene benchmark capability map: model x dataset x family accuracy."""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "scene_eval"
FAMS = ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]
MODELS = ["qwen3-vl-8b", "glm-4.1v-9b", "qwen2.5-vl-7b"]


def acc(recs, ds, fam):
    sub = [r for r in recs if r["dataset"] == ds and r["family"] == fam]
    if not sub:
        return None
    return sum(r["correct"] for r in sub) / len(sub), len(sub)


lines = []
for mk in MODELS:
    recs = [json.loads(l) for l in open(OUT / f"scene_{mk}.jsonl", encoding="utf-8")]
    lines.append(f"\n===== {mk} =====")
    lines.append(f"{'dataset':10s} " + " ".join(f"{f:>18s}" for f in FAMS))
    for ds in ["codd", "cdw_seg"]:
        row = [f"{ds:10s}"]
        for fam in FAMS:
            r = acc(recs, ds, fam)
            row.append(f"{r[0]:.3f} (n={r[1]})" if r else "—".rjust(18))
            if r:
                row[-1] = f"{r[0]:.3f} ({r[1]})".rjust(18)
        lines.append(" ".join(row))
    # F3 细分：cdw_seg F3 各模型（主战场）
    f3 = [r for r in recs if r["dataset"] == "cdw_seg" and r["family"] == "F3_dominance"]
    if f3:
        lines.append(f"  F3_dominance @cdw_seg: acc={sum(r['correct'] for r in f3) / len(f3):.3f} (n={len(f3)})")
        # 失败样例
        bad = [r for r in f3 if not r["correct"]][:6]
        for r in bad:
            lines.append(f"    GT={r['answer_text']:15s} 答={r['raw'].strip()[:40]}")

text = "\n".join(lines)
print(text)
(OUT / "scene_capability_map.txt").write_text(text, encoding="utf-8")
