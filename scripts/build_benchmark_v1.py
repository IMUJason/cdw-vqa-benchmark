"""Build benchmark v1 scene-level QA: CDW-Seg (430) + CODD sample (500).
F1 presence / F2 extent / F3 dominance / F4 contamination per image."""
import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml, class_stats, CLASSES as CODD_CLASSES
from src.cdw_vqa.cdwseg import CdwSeg, NON_MINERAL

CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
SEG = Path(r"datasets/cdw_seg/Ground_Truths_COCO_Format")
DATA = Path(__file__).resolve().parents[1] / "data"
random.seed(20261011)

CODD_NON_MINERAL = {"wood", "plastic", "foam", "pipes", "general_w"}
SEG_CLASSES = ["cardboard", "fill_dirt", "wood", "hard_plastic", "soft_plastic",
               "steel", "fabric", "concrete", "plasterboard"]


def mcq(options, answer, n=4):
    opts = [answer] + [o for o in options if o != answer]
    random.shuffle(opts)
    opts = opts[:n]
    if answer not in opts:
        opts[0] = answer
    letters = "ABCD"
    return {letters[i]: o for i, o in enumerate(opts)}, letters[opts.index(answer)]


def scene_items(dataset, image_id, stats, classes, non_mineral, split):
    present = [c for c in classes if c in stats]
    absent = [c for c in classes if c not in stats]
    items = []
    base = dict(dataset=dataset, image_id=str(image_id), level="scene", split=split)
    if present and absent:
        for c, truth in [(random.choice(present), "Yes"), (random.choice(absent), "No")]:
            items.append(dict(**base, family="F1_presence", qtype="yesno",
                              question=f"Is there any {c.replace('_',' ')} visible in this waste image?",
                              answer=truth))
    k = len(present)
    if k >= 1:
        opts, key = mcq([str(x) for x in {k - 1, k + 1, k + 2, max(1, k - 2)}], str(k))
        items.append(dict(**base, family="F2_extent", qtype="mcq",
                          question="How many distinct waste material classes are visible in this image?",
                          choices=opts, answer=key, answer_text=str(k)))
    if len(present) >= 3:
        dom = max(stats.items(), key=lambda kv: kv[1][2])[0]
        opts, key = mcq([c.replace("_", " ") for c in classes if c != dom],
                        dom.replace("_", " "))
        items.append(dict(**base, family="F3_dominance", qtype="mcq",
                          question="Which material dominates this waste load by visible area?",
                          choices=opts, answer=key, answer_text=dom))
    has_nm = bool(set(present) & non_mineral)
    items.append(dict(**base, family="F4_contamination", qtype="yesno",
                      question="Does this waste load contain non-mineral materials such as wood, plastic, fabric, or cardboard?",
                      answer="Yes" if has_nm else "No"))
    return items


def main():
    qa = []
    seg = CdwSeg(SEG)
    seg_ids = sorted(seg.images)
    for i, iid in enumerate(seg_ids):
        split = "train" if i % 10 < 7 else ("val" if i % 10 < 8.5 else "test")
        for it in scene_items("cdw_seg", f"2022_{iid + 1:04d}", seg.stats(iid),
                              SEG_CLASSES, NON_MINERAL, split):
            it["image_path"] = str(seg.path(iid))
            qa.append(it)
    codd_xmls = []
    for sp in ["training", "validation", "testing"]:
        codd_xmls += sorted(CODD.joinpath(sp).glob("*.xml"))
    random.shuffle(codd_xmls)
    for xp in codd_xmls[:500]:
        ci = parse_xml(xp)
        for it in scene_items("codd", ci.image_id, class_stats(ci),
                              CODD_CLASSES, CODD_NON_MINERAL, xp.parent.name):
            it["image_path"] = str(ci.path)
            qa.append(it)
    with open(DATA / "benchmark_v1_scene_qa.jsonl", "w", encoding="utf-8") as f:
        for it in qa:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print("total:", len(qa))
    print(Counter(f"{it['dataset']}_{it['family']}" for it in qa))
    print("DONE")


if __name__ == "__main__":
    main()
