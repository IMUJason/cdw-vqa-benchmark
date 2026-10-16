"""W2-2: build diagnostic v2 — 500 instances (350 pipes orthogonal + 150 others),
four-arm images, paired object/material QA, asserts, sample sheets.

Composition rationale (post-freeze revision, documented):
- orthogonal tier must be >=45% of headline diagnostic (红队 A5 处置) AND
  needs n>=343 pairs for TOST ±5pp power (红队 A4 处置) -> 350 pipes.
- pipes material ground truth is NOT in CODD annotations (metal vs PVC):
  material-Q for pipes gets needs_manual_label=True pending manual pass.
"""
import sys, json, re, random, csv
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml, class_stats, CLASSES
from src.cdw_vqa.arms import (arm_l0, arm_l1_inpaint, arm_l2_blur, arm_l3_crop,
                              contact_sheet)

ROOT = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
ARMS_OUT = Path(r"datasets/plan3_arms")
DATA = Path(__file__).resolve().parents[1] / "data"
SHEETS = Path(__file__).resolve().parents[1] / "results" / "diagnostic_v2_sheets"
for d in [ARMS_OUT, DATA, SHEETS]:
    d.mkdir(parents=True, exist_ok=True)

TIER = {"brick": "entangled", "concrete": "entangled", "tile": "entangled",
        "stone": "entangled", "wood": "entangled", "gypsum_board": "partial",
        "pipes": "orthogonal", "plastic": "material-pure", "foam": "material-pure",
        "general_w": "na"}
MATERIAL_ANS = {"brick": "fired clay", "concrete": "concrete", "tile": "ceramic",
                "stone": "natural stone", "wood": "wood",
                "gypsum_board": "gypsum", "plastic": "plastics", "foam": "polymer foam",
                "general_w": "mixed general waste", "pipes": None}  # pipes: manual
SEED = 20261010
random.seed(SEED)

N_PIPES, N_OTHERS = 350, 15  # 350 pipes + 15 x 10 other classes = 500


def session_of(image_id: str) -> str:
    m = re.match(r"^([a-zA-Z]+)", image_id)
    return m.group(1) if m else image_id[:2]


def save_jpg(im, path: Path):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, path
    buf.tofile(str(path))


def loc_ref(inst, ci):
    cx = float(inst.polygon[:, 0].mean()) / ci.width
    cy = float(inst.polygon[:, 1].mean()) / ci.height
    col = "left" if cx < .33 else ("center" if cx < .66 else "right")
    row = "top" if cy < .33 else ("middle" if cy < .66 else "bottom")
    return f"the fragment at {row}-{col}"


def main():
    # 1) pool
    pool = defaultdict(list)
    for split in ["training", "validation", "testing"]:
        for xp in sorted(ROOT.joinpath(split).glob("*.xml")):
            ci = parse_xml(xp)
            ncls = len(class_stats(ci))
            for ti, inst in enumerate(ci.instances):
                area = float(cv2.contourArea(inst.polygon.astype("float32")))
                if area < 1500:
                    continue
                pool[inst.name].append(dict(image_id=ci.image_id, split_src=split,
                                            session=session_of(ci.image_id),
                                            inst_idx=ti, cls=inst.name,
                                            n_classes_in_image=ncls,
                                            area_px=round(area, 1)))
    # 2) compose v2
    manifest = []
    pipes_c = pool["pipes"]
    random.shuffle(pipes_c)
    clutter_p = [d for d in pipes_c if d["n_classes_in_image"] >= 3]
    plain_p = [d for d in pipes_c if d["n_classes_in_image"] < 3]
    take_p = (clutter_p[:N_PIPES // 2] + plain_p)[:N_PIPES]
    assert len(take_p) == N_PIPES, len(take_p)
    for cls in CLASSES:
        if cls == "pipes":
            continue
        c = pool[cls][:]; random.shuffle(c)
        manifest.extend(c[:N_OTHERS])
    manifest.extend(take_p)
    # 3) session-balanced splits within tier groups (70/15/15)
    random.shuffle(manifest)
    for d in manifest:
        d["tier"] = TIER[d["cls"]]
    for tier_group in [["orthogonal"], ["entangled"], ["partial"],
                       ["material-pure"], ["na"]]:
        grp = [d for d in manifest if d["tier"] in tier_group]
        sess = sorted({d["session"] for d in grp}, key=lambda s: random.random())
        # greedy bin-pack sessions into 3 buckets by count
        buckets = {"train": 0.0, "val": 0.0, "test": 0.0}
        target = {"train": .7, "val": .15, "test": .15}
        assign = {}
        cnt = defaultdict(lambda: defaultdict(int))
        for s in sess:
            sizes = {k: cnt[k]["n"] / max(len(grp), 1) for k in buckets}
            k = min(buckets, key=lambda k: sizes[k] - target[k] * 0 +
                    (sizes[k] - target[k]))
            assign[s] = k
            cnt[k]["n"] += sum(1 for d in grp if d["session"] == s)
        for d in grp:
            d["split"] = assign[d["session"]]
    for i, d in enumerate(manifest):
        d["diag_id"] = i
    by_id_img = {}
    # 4) arms + QA
    qa = []
    parsed_cache = {}
    for d in manifest:
        key = (d["split_src"], d["image_id"])
        if key not in parsed_cache:
            parsed_cache[key] = parse_xml(ROOT / d["split_src"] / f"{d['image_id']}.xml")
        ci = parsed_cache[key]
        inst = ci.instances[d["inst_idx"]]
        assert inst.name == d["cls"], (d, inst.name)
        img = cv2.imread(str(ci.path))
        mask = ci.mask_of(inst)
        loc = loc_ref(inst, ci)
        arm_fns = {"L0": lambda: arm_l0(img), "L1": lambda: arm_l1_inpaint(img, mask),
                   "L2": lambda: arm_l2_blur(img, mask), "L3": lambda: arm_l3_crop(img, mask)}
        for arm_name, fn in arm_fns.items():
            fp = ARMS_OUT / f"{d['diag_id']:03d}_{arm_name}.jpg"
            if not fp.exists():  # resume: skip already-built arms
                save_jpg(fn(), fp)
            ref = loc if arm_name == "L0" else "the isolated fragment"
            obj_q = dict(diag_id=d["diag_id"], arm=arm_name,
                         family="instance_object",
                         question=f"What type of waste object is {ref}?",
                         answer=d["cls"].replace("_", " "),
                         tier=d["tier"], cls=d["cls"], split=d["split"],
                         image=fp.name, needs_manual_label=False)
            mat_ans = MATERIAL_ANS[d["cls"]]
            mat_q = dict(diag_id=d["diag_id"], arm=arm_name,
                         family="instance_material",
                         question=f"What material is {ref} mainly made of?",
                         answer=mat_ans if mat_ans else "UNLABELED",
                         tier=d["tier"], cls=d["cls"], split=d["split"],
                         image=fp.name,
                         needs_manual_label=(mat_ans is None))
            qa.extend([obj_q, mat_q])
    with open(DATA / "diagnostic_v2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dict(seed=SEED, version="v2",
                       note="350 pipes (orthogonal quota+power) + 15x10 others; "
                            "supersedes diagnostic_500.json",
                       n=len(manifest), instances=manifest), f, ensure_ascii=False, indent=1)
    with open(DATA / "diagnostic_v2_qa.jsonl", "w", encoding="utf-8") as f:
        for it in qa:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # 5) manual-label CSV for pipes (L3 crops are the clearest view)
    with open(DATA / "pipes_manual_label.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diag_id", "image_L3", "material_label(metal/PVC/other)"])
        for d in manifest:
            if d["cls"] == "pipes":
                w.writerow([d["diag_id"], f"{d['diag_id']:03d}_L3.jpg", ""])
    # 6) sample contact sheets
    sample = random.sample(manifest, 10)
    for d in sample:
        panels, labels = [], []
        for arm_name, lb in [("L0", "L0"), ("L1", "L1"), ("L2", "L2"), ("L3", "L3")]:
            im = cv2.imdecode(np.fromfile(str(ARMS_OUT / f"{d['diag_id']:03d}_{arm_name}.jpg"),
                                          dtype=np.uint8), cv2.IMREAD_COLOR)
            panels.append(im); labels.append(f"{lb} {d['cls']}")
        save_jpg(contact_sheet(panels, labels), SHEETS / f"sheet_{d['diag_id']:03d}_{d['cls']}.jpg")
    # 7) asserts & report
    assert len(qa) == 2 * 4 * len(manifest)
    n_manual = sum(1 for it in qa if it["needs_manual_label"])
    print("manifest:", len(manifest), Counter(d["cls"] for d in manifest))
    print("tiers:", Counter(d["tier"] for d in manifest))
    print("splits:", Counter(d["split"] for d in manifest))
    print("qa:", len(qa), "| needs_manual_label:", n_manual)
    print("arms dir:", ARMS_OUT, "| sheets:", SHEETS)
    print("DONE")


if __name__ == "__main__":
    main()
