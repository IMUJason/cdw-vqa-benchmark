"""Build judge views for the labeling web app:
{id}_L0h.jpg (in-situ with red contour+bbox), {id}_L1.jpg (context removed,
copied), {id}_Z.jpg (full-resolution zoom crop)."""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml

DATA = Path(__file__).resolve().parents[1] / "data"
CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
ARMS = Path(r"datasets/plan3_arms")
JV = ARMS / "judge_view"
JV.mkdir(parents=True, exist_ok=True)

man = {d["diag_id"]: d for d in json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]}
todo = [int(r["diag_id"]) for r in csv.DictReader(open(DATA / "pipes_to_judge.csv", encoding="utf-8-sig"))]
audit = [int(r["diag_id"]) for r in csv.DictReader(open(DATA / "pipes_auto_accept.csv", encoding="utf-8-sig"))
         if r.get("audit_flag") == "AUDIT"]
ids = sorted(set(todo + audit))
print("judge views for", len(ids), "instances")

_cache = {}

def save(im, p: Path):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    buf.tofile(str(p))

for n, did in enumerate(ids):
    d = man[did]
    key = (d["split_src"], d["image_id"])
    if key not in _cache:
        _cache[key] = parse_xml(CODD / d["split_src"] / f"{d['image_id']}.xml")
    ci = _cache[key]
    inst = ci.instances[d["inst_idx"]]
    img = cv2.imread(str(ci.path))
    mask = ci.mask_of(inst)
    # L0h: contour + bbox overlay
    l0h = img.copy()
    pts = np.round(inst.polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(l0h, [pts], True, (0, 0, 255), 6)
    x0, x1 = int(inst.polygon[:, 0].min()), int(inst.polygon[:, 0].max())
    y0, y1 = int(inst.polygon[:, 1].min()), int(inst.polygon[:, 1].max())
    cv2.rectangle(l0h, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), (0, 0, 255), 4)
    save(l0h, JV / f"{did}_L0h.jpg")
    # L1: copy from arms
    l1 = cv2.imdecode(np.fromfile(str(ARMS / f"{did:03d}_L1.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
    save(l1, JV / f"{did}_L1.jpg")
    # Z: full-res zoom crop
    px, py = int((x1 - x0) * 0.15) + 10, int((y1 - y0) * 0.15) + 10
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(img.shape[1] - 1, x1 + px), min(img.shape[0] - 1, y1 + py)
    crop = img[cy0:cy1 + 1, cx0:cx1 + 1]
    mc = mask[cy0:cy1 + 1, cx0:cx1 + 1]
    z = np.full_like(crop, 128)
    z[mc > 0] = crop[mc > 0]
    save(z, JV / f"{did}_Z.jpg")
    if n % 40 == 0:
        print(n, flush=True)
print("DONE", JV)
