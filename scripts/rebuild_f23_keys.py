"""Deterministically rebuild F2/F3 answer keys (position-balanced, in-image
distractors) while keeping every other field — and all F1/F4 rows — verbatim.

Motivation (review F16-1/P0): the v1 mcq() shuffled a pool larger than 4 and
repaired a truncated-away truth into slot A, so F3 keys sat at A for 66.7-78.3%
of items and the trivial always-A policy outscored every system. This rebuild:
  * F3 distractors: classes present in the same image (excluding the dominant),
    padded with dataset classes when fewer than 3 are present (fallback flagged
    in the row as distractor_domain for disclosure);
  * truth slot: strict round-robin over A-D in file order -> exactly balanced;
  * F2: same balanced construction over its count-bin pool;
  * per-row distractor sampling seeded by uid -> fully reproducible.

Usage: python scripts/rebuild_f23_keys.py   (writes data/benchmark_v1_scene_qa.jsonl)
"""
import json, random, re, sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.cdw_vqa.codd import parse_xml, class_stats
from src.cdw_vqa.cdwseg import CdwSeg

CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
SEG = Path(r"datasets/cdw_seg/Ground_Truths_COCO_Format")

SEG_CLASSES = ["cardboard", "fill_dirt", "wood", "hard_plastic", "soft_plastic",
               "steel", "fabric", "concrete", "plasterboard"]
CODD_CLASSES = ["concrete", "brick", "tile", "wood", "gypsum_board",
                "foam", "general_w", "stone", "plastic", "pipes"]

old = [json.loads(l) for l in open(BASE / "data" / "benchmark_v1_scene_qa_v1biased.jsonl",
                                    encoding="utf-8")]
seg = CdwSeg(SEG)
_stats_cache = {}


def image_classes(dataset, image_id):
    key = (dataset, image_id)
    if key in _stats_cache:
        return _stats_cache[key]
    if dataset == "codd":
        # locate the xml in whichever split directory holds this id
        found = None
        for sp in ["training", "validation", "testing"]:
            p = CODD / sp / (image_id + ".xml")
            if p.exists():
                found = p
                break
        classes = set(class_stats(parse_xml(found))) if found else set()
    else:
        m = re.match(r"2022_(\d+)", image_id)
        iid = int(m.group(1)) - 1 if m else None
        classes = set(seg.stats(iid)) if iid is not None else set()
    _stats_cache[key] = classes
    return classes


def balanced_mcq(rng, pool, answer, slot, n=4):
    pool = list(dict.fromkeys(o for o in pool if o != answer))
    rng.shuffle(pool)
    chosen = pool[: n - 1]
    n_eff = len(chosen) + 1          # 3-option rows shrink n; slot adapts
    if slot >= n_eff:
        slot = slot % n_eff
    opts = chosen[:slot] + [answer] + chosen[slot:]
    letters = "ABCD"
    return ({letters[i]: o for i, o in enumerate(opts)},
            letters[slot], len(pool))


rot = {"F2_extent": 0, "F3_dominance": 0}
out = []
n_fallback = 0
for r in old:
    if r["family"] in ("F1_presence", "F4_contamination"):
        out.append(r)
        continue
    row_id = f'{r["dataset"]}|{r["image_id"]}'
    rng = random.Random("cdwvqa-key|" + row_id + "|" + r["family"])
    if r["family"] == "F2_extent":
        k = int(r["choices"][r["answer"]])
        pool = [str(x) for x in {k - 1, k + 1, k + 2, max(1, k - 2)} if x != k]
        slot = rot["F2_extent"] % 4; rot["F2_extent"] += 1
        choices, key, _ = balanced_mcq(rng, pool, str(k), slot)
        nr = dict(r); nr["choices"] = choices; nr["answer"] = key
        out.append(nr)
    else:  # F3_dominance
        dom = r["choices"][r["answer"]]
        present = image_classes(r["dataset"], r["image_id"])
        in_img = [c for c in sorted(present) if c != dom]
        allc = CODD_CLASSES if r["dataset"] == "codd" else SEG_CLASSES
        fallback = [c for c in allc if c != dom and c not in present]
        used_fallback = len(in_img) < 3
        pool = [c.replace("_", " ") for c in (in_img if not used_fallback
                                              else in_img + fallback)]
        slot = rot["F3_dominance"] % 4; rot["F3_dominance"] += 1
        choices, key, pool_sz = balanced_mcq(rng, pool, dom.replace("_", " "), slot)
        if used_fallback:
            n_fallback += 1
        nr = dict(r); nr["choices"] = choices; nr["answer"] = key
        if used_fallback:
            nr["distractor_domain"] = "image+dataset"
        out.append(nr)

with open(BASE / "data" / "benchmark_v1_scene_qa.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
for fam in ["F2_extent", "F3_dominance"]:
    for ds in ["codd", "cdw_seg"]:
        c = Counter(r["answer"] for r in out if r["family"] == fam and r["dataset"] == ds)
        n = sum(c.values())
        print(f"{fam[:2]}/{ds}: n={n} " + " ".join(f"{L}={c.get(L,0)/n:.3f}" for L in "ABCD"))
print("F3 items using dataset-fallback distractors:", n_fallback)
print("total rows:", len(out))
