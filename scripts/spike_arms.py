"""Spike 2: generate four-arm panels for sample CODD instances + contact sheets."""
import sys, random
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml
from src.cdw_vqa.arms import (arm_l0, arm_l1_inpaint, arm_l1_fill, arm_l2_blur,
                              arm_l3_crop, contact_sheet)

ROOT = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
OUT = Path(__file__).resolve().parents[1] / "results" / "spike_arms"
OUT.mkdir(parents=True, exist_ok=True)


def save(im, path: Path):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok, path
    buf.tofile(str(path))  # unicode-path safe on Windows

random.seed(20261010)
xmls = sorted((ROOT / "training").glob("*.xml"))
random.shuffle(xmls)

picked = []  # (xml_path, target_index)
for xp in xmls:
    ci = parse_xml(xp)
    if len(ci.instances) < 4:
        continue
    # mid-size target to make context removal meaningful
    areas = sorted(range(len(ci.instances)),
                   key=lambda i: -cv2.contourArea(ci.instances[i].polygon.astype('float32')))
    picked.append((xp, areas[len(areas) // 2]))
    if len(picked) >= 6:
        break

for n, (xp, ti) in enumerate(picked):
    ci = parse_xml(xp)
    inst = ci.instances[ti]
    img = cv2.imread(str(ci.path))
    m = ci.mask_of(inst)
    panels = [
        (arm_l0(img), "L0 in-situ"),
        (arm_l1_inpaint(img, m), "L1 inpaint"),
        (arm_l1_fill(img, m), "L1b mean-fill"),
        (arm_l2_blur(img, m), "L2 bg-blur"),
        (arm_l3_crop(img, m), "L3 crop-matched"),
    ]
    for im, lb in panels:
        save(im, OUT / f"s{n}_{inst.name}_{lb.split()[0]}.jpg")
    sheet = contact_sheet([p for p, _ in panels], [lb for _, lb in panels])
    save(sheet, OUT / f"sheet_{n}_{inst.name}.jpg")
    print(f"[{n}] {ci.image_id} target={inst.name} n_inst={len(ci.instances)}")
print("DONE", OUT)
