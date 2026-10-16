"""Finalize pipes material ground truth:
1) merge human labels (priority) + audit-clean PVC auto-accepts;
2) consistency check against user's domain heuristics (white/green/red->PVC,
   corrugated->PVC, black->mixed) using masked color features;
3) update diagnostic_v2_qa.jsonl (pipes material-Q: UNLABELED -> label or excluded);
4) write report + flagged list for optional spot-check."""
import csv, json, sys, colorsys
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")

human = {int(r["diag_id"]): r["label"] for r in
         csv.DictReader(open(DATA / "pipes_human_labels.csv", encoding="utf-8-sig"))
         if r["diag_id"].isdigit() and r["label"] in ("metal", "PVC", "other")}
auto = {int(r["diag_id"]): r["auto_label"] for r in
        csv.DictReader(open(DATA / "pipes_auto_accept.csv", encoding="utf-8-sig"))}

final, source = {}, {}
for did, lab in human.items():
    final[did], source[did] = lab, "human"
for did, lab in auto.items():
    if did not in final:
        assert lab == "PVC", (did, lab)  # 金属项应已全部被人审覆盖
        final[did], source[did] = lab, "auto_pvc_audited"

man = {d["diag_id"]: d for d in json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]}
pipes_ids = [did for did, d in man.items() if d["cls"] == "pipes"]
unlabeled = [d for d in pipes_ids if d not in final]
print("final:", len(final), Counter(final.values()), "| unlabeled pipes:", len(unlabeled))

# ---- heuristic consistency check ----
_cache = {}
def color_feats(did):
    d = man[did]
    key = (d["split_src"], d["image_id"])
    if key not in _cache:
        _cache[key] = parse_xml(CODD / d["split_src"] / f"{d['image_id']}.xml")
    ci = _cache[key]
    img = cv2.imread(str(ci.path))
    m = ci.mask_of(ci.instances[d["inst_idx"]]) > 0
    px = img[m]
    b, g, r = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    hsv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return dict(v=hsv[2], s=hsv[1], h=hsv[0] * 360, r=r, g=g, b=b)

flagged = []
for did, lab in final.items():
    f = color_feats(did)
    hue, sat, val = f["h"], f["s"], f["v"]
    bright_neutral = val > 150 and sat < 0.25            # white-ish
    greenish = 60 < hue < 170 and sat > 0.15
    reddish = (hue < 25 or hue > 330) and sat > 0.25
    very_dark = val < 70                                  # black-ish
    rule_pvc = bright_neutral or greenish or reddish
    if lab == "metal" and rule_pvc:
        flagged.append((did, lab, "白/绿/红色法则→疑 PVC", f"v={val:.0f},s={sat:.2f},h={hue:.0f}"))
    if lab == "PVC" and very_dark:
        flagged.append((did, lab, "深色但标 PVC→可复核(黑色有概率金属)", f"v={val:.0f}"))
print("flagged:", len(flagged))

with open(DATA / "pipes_material_final.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["diag_id", "material", "source"])
    for did in sorted(final):
        w.writerow([did, final[did], source[did]])
with open(DATA / "pipes_flagged_review.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["diag_id", "current_label", "flag_reason", "color_feats"])
    w.writerows(flagged)

# ---- update QA ----
qa = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
n_fix = n_exc = 0
for it in qa:
    if it["needs_manual_label"]:
        lab = final.get(it["diag_id"])
        if lab == "other":
            it["excluded"] = True
            it["answer"] = "other (excluded from scoring)"
            n_exc += 1
        elif lab:
            it["answer"] = lab
            it["needs_manual_label"] = False
            n_fix += 1
        else:
            it["excluded"] = True  # 未标则排除
with open(DATA / "diagnostic_v2_qa.jsonl", "w", encoding="utf-8") as f:
    for it in qa:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"QA updated: labeled={n_fix}, excluded={n_exc}")
print("DONE")
