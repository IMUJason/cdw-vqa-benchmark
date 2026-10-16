"""Spike 3: QA pilot — template generation (F1-F4 scene-level + instance pairs)
+ Kimi paraphrase of 20 items. Outputs JSONL + paraphrase pairs + report."""
import sys, json, random, os
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.codd import parse_xml, class_stats, CLASSES

ROOT = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
OUT = Path(__file__).resolve().parents[1] / "results" / "spike_qa"
OUT.mkdir(parents=True, exist_ok=True)
SECRETS = Path(r"secrets.env")

random.seed(20261010)

# class -> (material, entailment_tier)
MATERIAL = {
    "brick": ("fired clay (brick)", "entangled"),
    "concrete": ("concrete", "entangled"),
    "tile": ("ceramic (tile)", "entangled"),
    "stone": ("natural stone", "entangled"),
    "wood": ("wood/timber", "entangled"),
    "gypsum_board": ("gypsum (plasterboard)", "partial"),
    "pipes": ("metal or PVC (pipe material — ambiguous)", "orthogonal"),
    "plastic": ("plastics", "material-pure"),
    "foam": ("polymer foam", "material-pure"),
    "general_w": ("general mixed waste", "na"),
}
NON_MINERAL = {"wood", "plastic", "foam", "pipes", "general_w"}


def mcq(options, answer, n=4):
    opts = [answer] + [o for o in options if o != answer]
    random.shuffle(opts)
    opts = opts[:n]
    if answer not in opts:
        opts[0] = answer
    letters = "ABCD"
    return {letters[i]: o for i, o in enumerate(opts)}, letters[opts.index(answer)]


def scene_qa(ci):
    stats = class_stats(ci)
    present = [c for c in CLASSES if c in stats]
    absent = [c for c in CLASSES if c not in stats]
    items = []
    # F1 presence
    for c, truth in [(random.choice(present), "Yes"),
                     (random.choice(absent), "No")]:
        items.append(dict(family="F1_presence", qtype="yesno",
                          question=f"Is there any {c.replace('_',' ')} visible in this waste load?",
                          answer=truth))
    # F2 extent
    k = len(present)
    opts, key = mcq([str(x) for x in {k - 1, k + 1, k + 2, max(1, k - 2)}], str(k))
    items.append(dict(family="F2_extent", qtype="mcq",
                      question="How many distinct material classes are visible on the belt?",
                      choices=opts, answer=key, answer_text=str(k)))
    # F3 dominance (categorical only; needs >=3 present classes for a real choice)
    if len(present) >= 3:
        dom = max(stats.items(), key=lambda kv: kv[1][2])[0]
        opts, key = mcq([c.replace("_", " ") for c in CLASSES if c != dom],
                        dom.replace("_", " "))
        items.append(dict(family="F3_dominance", qtype="mcq",
                          question="Which material dominates this load by visible area?",
                          choices=opts, answer=key, answer_text=dom))
    # F4 contamination
    has_nm = bool(NON_MINERAL & set(present))
    items.append(dict(family="F4_contamination", qtype="yesno",
                      question="Does this load contain non-mineral materials (e.g., wood, plastic, foam)?",
                      answer="Yes" if has_nm else "No"))
    return items


def instance_pairs(ci, ti):
    inst = ci.instances[ti]
    mat, tier = MATERIAL[inst.name]
    ys, xs = inst.polygon[:, 1], inst.polygon[:, 0]
    cx, cy = xs.mean() / ci.width, ys.mean() / ci.height
    col = "left" if cx < .33 else ("center" if cx < .66 else "right")
    row = "top" if cy < .33 else ("middle" if cy < .66 else "bottom")
    loc = f"{row}-{col}"
    pairs = []
    for arm in ["L0", "L1", "L3"]:
        ref = f"the fragment at {loc}" if arm == "L0" else "the isolated fragment"
        pairs.append(dict(family="F1_instance_object", arm=arm, qtype="short",
                          question=f"What type of waste object is {ref}?",
                          answer=inst.name.replace("_", " "), entailment_tier=tier))
        pairs.append(dict(family="F1_instance_material", arm=arm, qtype="short",
                          question=f"What material is {ref} mainly made of?",
                          answer=mat, entailment_tier=tier))
    return pairs


def kimi_paraphrase(items):
    from openai import OpenAI
    kv = dict(l.strip().split("=", 1) for l in SECRETS.read_text().splitlines() if "=" in l)
    client = OpenAI(api_key=kv["KIMI_API_KEY"], base_url=kv["KIMI_BASE_URL"] + "/v1")
    out = []
    for it in items:
        prompt = (f"Rewrite this waste-audit question so it sounds like a field auditor "
                  f"asking about a construction-waste sorting belt photo. Keep meaning and "
                  f"expected answer unchanged. Reply with the rewritten question only.\n\n"
                  f"Original: {it['question']}")
        r = client.chat.completions.create(
            model="k3", max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        para = r.choices[0].message.content.strip()
        out.append({"original": it["question"], "paraphrase": para})
        print("  para:", para[:80])
    return out


def main():
    xmls = sorted((ROOT / "training").glob("*.xml"))
    random.shuffle(xmls)
    qa = []
    for xp in xmls[:50]:
        ci = parse_xml(xp)
        for it in scene_qa(ci):
            it.update(image_id=ci.image_id, level="scene", source="template")
            qa.append(it)
    # instance pairs from 6 diverse images
    pairs = []
    used = set()
    for xp in xmls[60:90]:
        ci = parse_xml(xp)
        if len(ci.instances) < 3:
            continue
        areas = sorted(range(len(ci.instances)),
                       key=lambda i: -cv2.contourArea(ci.instances[i].polygon.astype("float32")))
        ti = areas[len(areas) // 2]
        if ci.instances[ti].name in used:
            continue
        used.add(ci.instances[ti].name)
        for it in instance_pairs(ci, ti):
            it.update(image_id=ci.image_id, level="instance", source="template")
            pairs.append(it)
        if len(used) >= 6:
            break
    qa.extend(pairs)
    with open(OUT / "pilot_100.jsonl", "w", encoding="utf-8") as f:
        for it in qa:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"total QA: {len(qa)} (scene {len(qa)-len(pairs)}, instance {len(pairs)})")
    if not os.environ.get("SKIP_KIMI"):
        sample = random.sample([it for it in qa if it["level"] == "scene"], 14) + \
                 [it for it in pairs if it["family"] == "F1_instance_material"][:6]
        paras = kimi_paraphrase(sample)
        with open(OUT / "paraphrase_pairs.json", "w", encoding="utf-8") as f:
            json.dump(paras, f, ensure_ascii=False, indent=1)
    print("DONE", OUT)


if __name__ == "__main__":
    main()
