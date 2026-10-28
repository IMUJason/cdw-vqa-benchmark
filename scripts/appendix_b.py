"""Generate Appendix B (pipes material confusion) from released responses.

Canonical two-level classifier exactly as specified in Section 3.7:
  check order: fine PVC -> fine metal -> coarse plastic-family -> other;
  matching is word-level containment on the cleaned answer.
Usage: python scripts/appendix_b.py
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

FINE_PVC = re.compile(r"\b(pvc|upvc|cpvc|vinyl|polyvinyl\s+chloride)\b")
FINE_MET = re.compile(r"\b(metal|metallic|steel|iron|copper|aluminum|aluminium|brass|galvanis?zed)\b")
COARSE_PVC = re.compile(r"\b(plastic|plastics|polymer|polypropylene|polyethylene)\b")


def clean(a: str) -> str:
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
    a = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</?answer>|\*+", "", a)
    return a.strip().lower()


def cls(raw: str) -> str:
    a = clean(raw)
    if FINE_PVC.search(a):
        return "PVC"
    if FINE_MET.search(a):
        return "metal"
    if COARSE_PVC.search(a):
        return "plastic-only"
    return "other"


ROWS = [("zs Qwen3-VL", "results/w6/responses_qwen3-vl-8b.jsonl"),
        ("zs Qwen2.5-VL", "results/w6/responses_qwen2.5-vl-7b.jsonl"),
        ("zs GLM-4.1V", "results/w6/responses_glm-4.1v-9b.jsonl"),
        ("ad Qwen3-VL", "results/adapted/adapted_qwen3vl_v2.jsonl"),
        ("ad Qwen2.5-VL", "results/adapted/adapted_qwen25vl_v2.jsonl")]

if __name__ == "__main__":
    for name, p in ROWS:
        rows = [json.loads(l) for l in open(BASE / p, encoding="utf-8")]
        for arm in ["L0", "L1"]:
            sel = [r for r in rows if r.get("family") == "instance_material"
                   and r.get("arm") == arm and r.get("cls") == "pipes"
                   and not r.get("needs_manual_label")
                   and str(r.get("split", "")).lower() in ("test", "testing")]  # test split
            counts = {"PVC": 0, "plastic-only": 0, "metal": 0, "other": 0}
            for r in sel:
                counts[cls(r["raw"])] += 1
            print(f"{name} {arm}: {counts['PVC']}/{counts['plastic-only']}/{counts['metal']}/{counts['other']}")
