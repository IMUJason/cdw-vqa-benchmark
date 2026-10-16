"""W2-1: freeze the 500-instance diagnostic manifest.
Stratified 50/class x 10 classes; session-level split via filename prefixes;
records entailment tier and clutter level for each instance."""
import sys, json, re, random
from pathlib import Path
from collections import defaultdict
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml, class_stats, CLASSES

ROOT = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(parents=True, exist_ok=True)

TIER = {"brick": "entangled", "concrete": "entangled", "tile": "entangled",
        "stone": "entangled", "wood": "entangled", "gypsum_board": "partial",
        "pipes": "orthogonal", "plastic": "material-pure", "foam": "material-pure",
        "general_w": "na"}
random.seed(20261010)

def session_of(image_id: str) -> str:
    m = re.match(r"^([a-zA-Z]+)", image_id)
    return m.group(1) if m else image_id[:2]

def main():
    pool = []  # candidate dicts
    for split in ["training", "validation", "testing"]:
        for xp in sorted((ROOT / split).glob("*.xml")):
            ci = parse_xml(xp)
            stats = class_stats(ci)
            ncls = len(stats)
            for ti, inst in enumerate(ci.instances):
                area = float(cv2.contourArea(inst.polygon.astype("float32")))
                if area < 1500:  # skip slivers
                    continue
                pool.append(dict(image_id=ci.image_id, split_src=split,
                                 session=session_of(ci.image_id),
                                 inst_idx=ti, cls=inst.name,
                                 area_px=round(area, 1),
                                 n_classes_in_image=ncls,
                                 tier=TIER.get(inst.name, "na")))
    print("pool:", len(pool))
    by_cls = defaultdict(list)
    for d in pool:
        by_cls[d["cls"]].append(d)
    # session-level split: assign sessions to train/val/test (70/15/15)
    sessions = sorted({d["session"] for d in pool})
    random.shuffle(sessions)
    n = len(sessions)
    sess_split = {}
    for i, s in enumerate(sessions):
        sess_split[s] = "train" if i < 0.7 * n else ("val" if i < 0.85 * n else "test")
    # stratified pick 50/class: prefer cluttered images (ncls>=3) up to 60%,
    # span area sizes; keep class balance across splits when possible
    manifest = []
    for cls in CLASSES:
        cands = by_cls.get(cls, [])
        random.shuffle(cands)
        clutter = [d for d in cands if d["n_classes_in_image"] >= 3]
        plain = [d for d in cands if d["n_classes_in_image"] < 3]
        take = clutter[:30] + plain[:50 - len(clutter[:30])]
        if len(take) < 50:
            take += (clutter[30:] + plain[50 - len(clutter[:30]):])[:50 - len(take)]
        for d in take[:50]:
            d["split"] = sess_split[d["session"]]
            manifest.append(d)
        print(f"{cls}: pool={len(cands)} take={len(take[:50])}")
    random.shuffle(manifest)
    for i, d in enumerate(manifest):
        d["diag_id"] = i
    with open(OUT / "diagnostic_500.json", "w", encoding="utf-8") as f:
        json.dump(dict(seed=20261010, n=len(manifest), instances=manifest),
                  f, ensure_ascii=False, indent=1)
    from collections import Counter
    print("n:", len(manifest), Counter(d["split"] for d in manifest))
    print(Counter(d["cls"] for d in manifest))
    print(Counter(d["tier"] for d in manifest))
    print("DONE", OUT / "diagnostic_500.json")

if __name__ == "__main__":
    main()
