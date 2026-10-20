"""Oracle localization: draw red bbox on L0 test images for pipes, re-ask adapted
Qwen3-VL, compare with unboxed L0. If boxed >> unboxed, residual bottleneck is
reference/localization, not perception."""
import sys, json
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.cdw_vqa.codd import parse_xml

CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
ARMS = Path(r"datasets/plan3_arms")
OUT = ARMS / "oracle_box"
OUT.mkdir(exist_ok=True)
DATA = BASE / "data"

man = {d["diag_id"]: d for d in json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]}
pipes_test = [d for d in man.values() if d["cls"] == "pipes" and d["split"] == "test"]
print("test pipes:", len(pipes_test))

_cache = {}


def save(im, p):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    buf.tofile(str(p))


for d in pipes_test:
    key = (d["split_src"], d["image_id"])
    if key not in _cache:
        _cache[key] = parse_xml(CODD / d["split_src"] / f"{d['image_id']}.xml")
    ci = _cache[key]
    inst = ci.instances[d["inst_idx"]]
    img = cv2.imread(str(ci.path))
    x0, x1 = int(inst.polygon[:, 0].min()), int(inst.polygon[:, 0].max())
    y0, y1 = int(inst.polygon[:, 1].min()), int(inst.polygon[:, 1].max())
    ob = img.copy()
    cv2.rectangle(ob, (x0 - 12, y0 - 12), (x1 + 12, y1 + 12), (0, 0, 255), 10)
    save(ob, OUT / f"{d['diag_id']:03d}_L0box.jpg")
print("boxed images:", len(list(OUT.glob('*.jpg'))))
