"""Evaluate the trained segmentation baseline on the benchmark's scene questions.

Answers F1 (presence), F2 (extent/count), F3 (dominance), F4 (contamination)
from predicted masks via the unified ontology, under the same question files
and the same correct-checks as the MLLMs. Presence threshold tuned on CODD val.
Reports: CODD test (in-domain held-out) and CDW-Seg (cross-domain zero-shot,
all splits -- same items the zero-shot MLLM numbers cover).

Unified mapping (Table 2 of the paper):
  CODD preds: concrete->concrete; brick/stone/tile->mineral; wood->wood;
    gypsum_board->gypsum; plastic->plastics; foam->foam; pipes->metallike;
    general_w->(nothing)
  CDW-Seg question classes: timber->wood; plasterboard->gypsum;
    hard_plastic/soft_plastic->plastics; steel->metallike; concrete->concrete;
    fill_dirt/fabric/cardboard->(out of vocabulary: never predictable)
  CODD question classes: same as preds (brick/stone/tile->mineral etc.)
Non-mineral set for F4: wood, plastics, foam, metallike (mirrors the
benchmark generator's CODD_NON_MINERAL).
"""
import sys, json, re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.cdw_vqa.codd import parse_xml, CLASSES as CODD_CLASSES
from src.cdw_vqa.cdwseg import CdwSeg

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
SEG = Path(r"datasets/cdw_seg/Ground_Truths_COCO_Format")
SIZE = 448
CLS = sorted(CODD_CLASSES)
CIDX = {c: i + 1 for i, c in enumerate(CLS)}

UNI_OF_CODD = {"concrete": "concrete", "brick": "mineral", "stone": "mineral",
               "tile": "mineral", "wood": "wood", "gypsum_board": "gypsum",
               "plastic": "plastics", "foam": "foam", "pipes": "metallike"}
UNI_OF_SEGQ = {"concrete": "concrete", "timber": "wood", "plasterboard": "gypsum",
               "hard_plastic": "plastics", "soft_plastic": "plastics",
               "steel": "metallike"}
NON_MINERAL = {"wood", "plastics", "foam", "metallike"}


def uni_of_question_class(c):
    c = c.strip().replace(" ", "_")
    if c in UNI_OF_SEGQ:
        return UNI_OF_SEGQ[c]
    if c in UNI_OF_CODD:
        return UNI_OF_CODD[c]
    return None  # out of segmenter vocabulary


def predict(model, path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    x = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
    x = torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
    x = ((x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) /
         torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)).unsqueeze(0).cuda()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        out = model(x)["out"][0]
    prob = torch.softmax(out.float(), 0).cpu().numpy()
    areas = defaultdict(float)
    for c, i in CIDX.items():
        u = UNI_OF_CODD.get(c)
        if u:
            areas[u] += prob[i].sum() / (SIZE * SIZE)
    return areas, (h, w)


def answer(item, areas, tau):
    q = item["question"]
    fam = item["family"]
    if fam == "F1_presence":
        m = re.search(r"Is there any ([a-z_ ]+?) visible", q)
        u = uni_of_question_class(m.group(1))
        present = (u is not None) and areas.get(u, 0.0) >= tau
        return "Yes" if present else "No"
    if fam == "F4_contamination":
        present = any(areas.get(u, 0.0) >= tau for u in NON_MINERAL)
        return "Yes" if present else "No"
    if fam == "F2_extent":
        n = sum(1 for u in set(UNI_OF_CODD.values()) if areas.get(u, 0.0) >= tau)
        best, bd = None, 1e9
        for k, v in item["choices"].items():
            try:
                d = abs(int(v) - n)
            except ValueError:
                continue
            if d < bd:
                best, bd = k, d
        return best or "A"
    if fam == "F3_dominance":
        best_u = max(areas, key=areas.get) if areas else None
        for k, v in item["choices"].items():
            if uni_of_question_class(str(v)) == best_u:
                return k
        return "A"
    return None


def score_items(model, items, images, tau):
    rows = []
    cache = {}
    for it in items:
        key = (it["dataset"], it["image_id"])
        if key not in cache:
            cache[key] = predict(model, images[key])
        areas = cache[key][0]
        a = answer(it, areas, tau)
        ok = (a == it["answer"])
        rows.append(dict(it, pred=a, correct=bool(ok)))
    return rows


def main():
    model = deeplabv3_resnet50(num_classes=len(CLS) + 1)
    model.load_state_dict(torch.load(Path(r"models/seg_baseline/codd_deeplab.pt"),
                                     map_location="cpu"))
    model = model.cuda().eval()

    qa = [json.loads(l) for l in open(BASE / "data" / "benchmark_v1_scene_qa.jsonl",
                                      encoding="utf-8")]
    # image paths
    imgs = {}
    for it in qa:
        imgs[(it["dataset"], it["image_id"])] = it["image_path"]

    # threshold tuning on CODD val (presence+contamination only)
    val = [it for it in qa if it["dataset"] == "codd"
           and it["split"] in ("val", "validation")
           and it["family"] in ("F1_presence", "F4_contamination")]
    best_tau, best_acc = 0.005, -1
    for tau in [0.001, 0.005, 0.01, 0.02, 0.05]:
        rows = score_items(model, val, imgs, tau)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"tau={tau}: val acc {acc:.3f} (n={len(rows)})")
        if acc > best_acc:
            best_acc, best_tau = acc, tau
    print("chosen tau:", best_tau)

    out = {"tau": best_tau}
    rows_all = []
    for name, sel in [
        ("codd_test", lambda it: it["dataset"] == "codd" and it["split"] in ("test", "testing")),
        ("cdw_seg_all", lambda it: it["dataset"] == "cdw_seg"),
    ]:
        items = [it for it in qa if sel(it)]
        rows = score_items(model, items, imgs, best_tau)
        rows_all += rows
        print(f"\n== {name} (n={len(rows)})")
        out[name] = {}
        for fam in ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]:
            sub = [r for r in rows if r["family"] == fam]
            if not sub:
                continue
            acc = sum(r["correct"] for r in sub) / len(sub)
            out[name][fam] = dict(acc=round(acc, 3), n=len(sub))
            print(f"  {fam:16s} acc={acc:.3f} (n={len(sub)})")
    OUT = BASE / "results" / "seg_baseline"
    OUT.mkdir(exist_ok=True)
    with open(OUT / "seg_baseline.jsonl", "w", encoding="utf-8") as f:
        for r in rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "seg_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
