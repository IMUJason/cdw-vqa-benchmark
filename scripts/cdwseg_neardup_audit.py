"""Near-duplicate audit of the CDW-Seg index-modulo split (released audit).

For every CDW-Seg test image, finds the nearest training frame by 64x64 grey
mean absolute error on the 1280-px cached images, reports the near-duplicate
count under a threshold of 40, and re-scores the CDW-Seg test cells with those
test images excluded (zero-shot under the unified answer rule, adapted from the
stored flags) so the sensitivity claim in the paper can be checked.

Usage: python scripts/cdwseg_neardup_audit.py
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from rescore_scene import unified, hit  # noqa: E402

CACHE = Path("datasets/plan3_scene_cache")  # expected local layout; configure if needed
QA = BASE / "data" / "benchmark_v1_scene_qa.jsonl"
FAMS = ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]


def grey64(img_id):
    p = CACHE / (img_id + ".jpg")
    if not p.exists():
        return None
    im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (64, 64)).astype(np.float32)


def main():
    rows = [json.loads(l) for l in open(QA, encoding="utf-8")]
    split_of = {}
    for r in rows:
        if r["dataset"] == "cdw_seg":
            split_of[r["image_id"]] = {"training": "train", "validation": "val",
                                       "testing": "test"}.get(r["split"], r["split"])
    train = [(i, g) for i, g in ((i, grey64(i)) for i, s in split_of.items() if s == "train")
             if g is not None]
    test_ids = [i for i, s in split_of.items() if s == "test"]

    near, ranked = [], []
    for t in test_ids:
        gt = grey64(t)
        if gt is None:
            continue
        best = min((float(np.abs(gt - g).mean()), i) for i, g in train)
        ranked.append((best[0], t, best[1]))
        if best[0] < 40:
            near.append(t)
    ranked.sort()
    print(f"test images: {len(test_ids)}; near-duplicates (MAE<40): {len(near)}")
    for m, t, tr in ranked[:10]:
        print(f"  {t}: MAE {m:.1f} (nearest train {tr})")

    NEAR = set(near)

    def cells(path, scorer, is_scene):
        rws = [json.loads(l) for l in open(path, encoding="utf-8")]
        if is_scene:
            pass
        out = {}
        for fam in FAMS:
            sel = [r for r in rws if r["family"] == fam and r.get("dataset") == "cdw_seg"
                   and r.get("split") in ("test", "testing")]
            clean = [r for r in sel if r["image_id"] not in NEAR]
            if sel and clean:
                out[fam] = (sum(scorer(r) for r in sel) / len(sel),
                            sum(scorer(r) for r in clean) / len(clean), len(sel), len(clean))
        return out

    print("\nzero-shot CDW-Seg test cells (unified rule): all -> clean")
    for tag, fn in [("qwen3", "scene_qwen3-vl-8b.jsonl"), ("qwen25", "scene_qwen2.5-vl-7b.jsonl")]:
        p = BASE / "results" / "scene_eval" / fn
        if not p.exists():
            continue
        for fam, (a, b, n1, n2) in cells(p, lambda r: hit(r, unified(r)), True).items():
            print(f"  {tag} {fam[:2]}: {a:.3f} -> {b:.3f}  (n {n1}->{n2}, delta {b-a:+.3f})")

    print("\nadapted CDW-Seg test cells (stored correct): all -> clean")
    for tag, fn in [("qwen3", "adapted_qwen3vl_v2.jsonl"), ("qwen25", "adapted_qwen25vl_v2.jsonl")]:
        p = BASE / "results" / "adapted" / fn
        if not p.exists():
            continue
        for fam, (a, b, n1, n2) in cells(p, lambda r: bool(r.get("correct")), False).items():
            print(f"  {tag} {fam[:2]}: {a:.3f} -> {b:.3f}  (n {n1}->{n2}, delta {b-a:+.3f})")


if __name__ == "__main__":
    main()
