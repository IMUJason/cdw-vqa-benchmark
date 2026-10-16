"""W6 rescue analysis: error typology on raw answers.
The pre-registered accuracy-delta gate failed (with two scorer artifacts).
The CMB signature is in ANSWER SPACE: everyday-object terms hijacking
material questions. No GPU needed — works off responses_*.jsonl."""
import json, re
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "w6"
ARMS = ["L0", "L1", "L2", "L3"]

MAT_LEX = {"concrete", "cement", "brick", "clay", "ceramic", "stone", "rock",
           "wood", "timber", "gypsum", "plaster", "plasterboard", "drywall",
           "plastic", "plastics", "polymer", "pvc", "vinyl", "foam", "styrofoam",
           "metal", "steel", "iron", "copper", "aluminum", "glass", "tile",
           "cardboard", "paper", "fabric", "rubber", "dirt", "soil", "sand",
           "waste", "debris", "rubble", "marble", "granite"}
# everyday objects that are NOT waste-material answers = object-prior hijack
EVERYDAY = {"cigarette", "cigarettes", "pencil", "pencils", "bone", "bones",
            "crayon", "chalk", "candy", "candle", "toothbrush", "straw",
            "toothpick", "matchstick", "toy", "doll", "figurine", "sausage",
            "bread", "banana", "carrot", "cucumber", "cookie", "cracker",
            "soap", "sponge", "brush", "pen", "marker", "lipstick", " syringe",
            "pill", "capsule", "bullet", "shell", "seashell", "stick", "twig",
            "branch", "log", "wooden stick", "pipe cleaner", "q-tip"}
OBJ_LEX = {"pipe", "pipes", "piping", "tube", "hose", "brick", "tile",
           "board", "panel", "sheet", "block", "plank", "batten", "beam",
           "rod", "bar", "rebar", "wire", "cable", "insulation"}


def clean(a: str) -> str:
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
    a = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</?answer>", "", a)
    return a.strip().lower().strip(".,!\"' ")


def typology(item, raw):
    a = clean(raw)
    if item["correct"]:
        return "correct"
    toks = set(re.findall(r"[a-z]+", a))
    if toks & EVERYDAY or any(e in a for e in EVERYDAY):
        return "everyday_object_hijack"
    if toks & MAT_LEX or any(m in a for m in MAT_LEX):
        return "wrong_material"
    if toks & OBJ_LEX or any(o in a for o in OBJ_LEX):
        return "wrong_object"
    return "other"


def main():
    models = ["qwen3-vl-8b", "glm-4.1v-9b", "qwen2.5-vl-7b"]
    lines = []
    for mk in models:
        recs = [json.loads(l) for l in open(OUT / f"responses_{mk}.jsonl", encoding="utf-8")]
        mat_q = [r for r in recs if r["family"] == "instance_material" and r["tier"] == "orthogonal"]
        obj_q = [r for r in recs if r["family"] == "instance_object" and r["tier"] == "orthogonal"]
        lines.append(f"\n===== {mk} (pipes 正交档) =====")
        for fam, rec in [("material-Q", mat_q), ("object-Q", obj_q)]:
            lines.append(f"  [{fam}]")
            for arm in ARMS:
                sub = [r for r in rec if r["arm"] == arm]
                ty = Counter(typology(r, r["raw"]) for r in sub)
                n = len(sub)
                lines.append(
                    f"    {arm}: correct={ty['correct'] / n:.3f} "
                    f"wrong_mat={ty['wrong_material'] / n:.3f} "
                    f"wrong_obj={ty['wrong_object'] / n:.3f} "
                    f"**hijack={ty['everyday_object_hijack'] / n:.3f}** "
                    f"other={ty['other'] / n:.3f}")
        # 劫持示例
        ex = [r for r in mat_q if typology(r, r["raw"]) == "everyday_object_hijack"][:8]
        lines.append("  劫持样例(material-Q→日常物体词): " +
                     "; ".join(f"{r['arm']}:{clean(r['raw'])[:20]}" for r in ex))
    text = "\n".join(lines)
    print(text)
    (OUT / "w6_typology.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
