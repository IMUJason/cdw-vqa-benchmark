"""W6 gate run: 3 local models x 485 instances x 4 arms x paired questions.
Sequential per model, resume-safe (skips answered items).
Usage: python scripts/run_w6_gate.py [model_key ...]"""
import sys, json, re, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.cdw_vqa.models_api import LocalQwenVL

DATA = BASE / "data"
ARMS = Path(r"datasets/plan3_arms")
OUT = BASE / "results" / "w6"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen3-vl-8b": dict(path=r"models/qwen3-vl-8b-instruct"),
    "glm-4.1v-9b": dict(path=r"models/glm-4.1v-9b-thinking"),
    "qwen2.5-vl-7b": dict(path=r"models/qwen2.5-vl-7b-instruct"),
}

OBJ_ALIAS = {
    "brick": {"brick"}, "concrete": {"concrete"}, "tile": {"tile"},
    "stone": {"stone"}, "wood": {"wood", "timber"},
    "gypsum_board": {"gypsum board", "plasterboard", "drywall", "gypsum"},
    "pipes": {"pipe", "pipes", "piping", "tube", "hose"},
    "plastic": {"plastic", "plastics"},
    "foam": {"foam", "styrofoam"},
    "general_w": {"general waste", "mixed waste", "general"},
}
MAT_ALIAS = {
    "fired clay": {"fired clay", "clay", "brick", "ceramic brick", "earthenware"},
    "concrete": {"concrete", "cement"},
    "ceramic": {"ceramic", "tile", "ceramic tile", "porcelain"},
    "natural stone": {"stone", "natural stone", "rock"},
    "wood": {"wood", "timber", "lumber"},
    "gypsum": {"gypsum", "plasterboard", "drywall", "plaster", "gypsum board"},
    "plastics": {"plastic", "plastics", "polymer", "pvc"},
    "polymer foam": {"foam", "polymer foam", "styrofoam", "polyurethane"},
    "mixed general waste": {"mixed waste", "general waste", "mixed"},
    "metal": {"metal", "steel", "iron", "copper", "aluminum", "metallic"},
    "PVC": {"pvc", "plastic", "plastics", "polymer", "vinyl"},
}


def clean(a: str) -> str:
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
    a = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</?answer>", "", a)
    return a.strip().lower().strip(".,!\"' ")


def score(item, raw: str) -> bool:
    a = clean(raw)
    if item["family"] == "instance_object":
        aliases = OBJ_ALIAS[item["cls"]]
    else:
        aliases = MAT_ALIAS.get(item["answer"], {item["answer"].lower()})
    return any(al in a for al in aliases)


def main():
    keys = sys.argv[1:] or list(MODELS)
    qa = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
    qa = [it for it in qa if not it.get("excluded")]
    print(f"items: {len(qa)}")
    for mk in keys:
        conf = MODELS[mk]
        resp_path = OUT / f"responses_{mk}.jsonl"
        done = set()
        if resp_path.exists():
            for l in open(resp_path, encoding="utf-8"):
                r = json.loads(l)
                done.add((r["diag_id"], r["arm"], r["family"]))
        todo = [it for it in qa if (it["diag_id"], it["arm"], it["family"]) not in done]
        print(f"\n=== {mk}: todo {len(todo)} (done {len(done)})")
        if not todo:
            continue
        model = LocalQwenVL(conf["path"])
        t0 = time.time()
        with open(resp_path, "a", encoding="utf-8") as f:
            for i, it in enumerate(todo):
                img = ARMS / it["image"]
                q = it["question"] + " Answer with one word or a short phrase."
                try:
                    raw = model.ask(q, img, max_tokens=1024)
                except Exception as e:
                    raw = f"ERROR {type(e).__name__}: {e}"
                rec = dict(it, raw=raw, correct=score(it, raw))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 200 == 0:
                    el = time.time() - t0
                    print(f"{mk} {i}/{len(todo)} {el / (i + 1):.2f}s/it", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
