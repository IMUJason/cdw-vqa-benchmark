"""Round-2 setup: audit caught systematic metal->PVC bias in auto-accept.
All non-audited metal auto-accepts go back to human. Build missing judge
views, write label_items_round2.json, clean test row from human labels."""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
ARMS = Path(r"datasets/plan3_arms")
JV = ARMS / "judge_view"

human = {int(r["diag_id"]): r["label"] for r in csv.DictReader(open(DATA / "pipes_human_labels.csv", encoding="utf-8-sig"))
         if r["label"] != "test_delete_me" and r["diag_id"].isdigit()}
auto = list(csv.DictReader(open(DATA / "pipes_auto_accept.csv", encoding="utf-8-sig")))
round2 = [int(r["diag_id"]) for r in auto
          if r["auto_label"] == "metal" and int(r["diag_id"]) not in human]
print("round2 items:", len(round2))

# clean human labels (drop test row)
with open(DATA / "pipes_human_labels.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["diag_id", "label", "ts"])
    for did, lab in sorted(human.items()):
        w.writerow([did, lab, ""])

# build views for round2 ids not yet present
man = {d["diag_id"]: d for d in json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]}
_cache = {}

def save(im, p):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92]); assert ok
    buf.tofile(str(p))

made = 0
for did in round2:
    if (JV / f"{did}_L0h.jpg").exists():
        continue
    d = man[did]
    key = (d["split_src"], d["image_id"])
    if key not in _cache:
        _cache[key] = parse_xml(CODD / d["split_src"] / f"{d['image_id']}.xml")
    ci = _cache[key]
    inst = ci.instances[d["inst_idx"]]
    img = cv2.imread(str(ci.path)); mask = ci.mask_of(inst)
    l0h = img.copy()
    pts = np.round(inst.polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(l0h, [pts], True, (0, 0, 255), 6)
    x0, x1 = int(inst.polygon[:, 0].min()), int(inst.polygon[:, 0].max())
    y0, y1 = int(inst.polygon[:, 1].min()), int(inst.polygon[:, 1].max())
    cv2.rectangle(l0h, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), (0, 0, 255), 4)
    save(l0h, JV / f"{did}_L0h.jpg")
    l1 = cv2.imdecode(np.fromfile(str(ARMS / f"{did:03d}_L1.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
    save(l1, JV / f"{did}_L1.jpg")
    px, py = int((x1 - x0) * .15) + 10, int((y1 - y0) * .15) + 10
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(img.shape[1] - 1, x1 + px), min(img.shape[0] - 1, y1 + py)
    crop = img[cy0:cy1 + 1, cx0:cx1 + 1]; mc = mask[cy0:cy1 + 1, cx0:cx1 + 1]
    z = np.full_like(crop, 128); z[mc > 0] = crop[mc > 0]
    save(z, JV / f"{did}_Z.jpg")
    made += 1
print("views built:", made)

json.dump([{"id": i, "audit": False} for i in sorted(round2)],
          open(DATA / "label_items_round2.json", "w"))
print("round2 json written")
