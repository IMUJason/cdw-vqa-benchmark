"""Build judge contact sheets: 20 zoom crops per page with diag_id labels,
for fast human adjudication. Human view uses full-resolution zoom crops
(the L3 resolution-matching constraint binds model arms, not human review)."""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml

DATA = Path(__file__).resolve().parents[1] / "data"
CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
OUT = Path(__file__).resolve().parents[1] / "results" / "judge_sheets"
OUT.mkdir(parents=True, exist_ok=True)
CELL = 300
COLS, ROWS = 5, 4

_man = {d["diag_id"]: d for d in json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]}
_cache = {}


def zoom_crop(did: int) -> np.ndarray:
    d = _man[did]
    key = (d["split_src"], d["image_id"])
    if key not in _cache:
        _cache[key] = parse_xml(CODD / d["split_src"] / f"{d['image_id']}.xml")
    ci = _cache[key]
    inst = ci.instances[d["inst_idx"]]
    img = cv2.imread(str(ci.path))
    mask = ci.mask_of(inst)
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x1 - x0) * 0.15) + 10, int((y1 - y0) * 0.15) + 10
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(img.shape[1] - 1, x1 + px), min(img.shape[0] - 1, y1 + py)
    crop = img[y0:y1 + 1, x0:x1 + 1]
    mc = mask[y0:y1 + 1, x0:x1 + 1]
    out = np.full_like(crop, 128)
    out[mc > 0] = crop[mc > 0]
    return out


def make_sheet(ids, out_path):
    grid = np.full((ROWS * CELL, COLS * CELL, 3), 255, np.uint8)
    for i, did in enumerate(ids):
        cell = cv2.resize(zoom_crop(int(did)), (CELL, CELL))
        cv2.putText(cell, str(did), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 255), 3)
        r, c = divmod(i, COLS)
        grid[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = cell
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    buf.tofile(str(out_path))


todo = [r["diag_id"] for r in csv.DictReader(open(DATA / "pipes_to_judge.csv", encoding="utf-8-sig"))]
for p in range(0, len(todo), COLS * ROWS):
    make_sheet(todo[p:p + COLS * ROWS], OUT / f"judge_page_{p // (COLS * ROWS) + 1:02d}.jpg")
audit = [r["diag_id"] for r in csv.DictReader(open(DATA / "pipes_auto_accept.csv", encoding="utf-8-sig"))
         if r.get("audit_flag") == "AUDIT"]
make_sheet(audit, OUT / "audit_page.jpg")
print(f"judge pages: {(len(todo) + 19) // 20}, audit page: 1, total ids: {len(todo)}+{len(audit)}")
