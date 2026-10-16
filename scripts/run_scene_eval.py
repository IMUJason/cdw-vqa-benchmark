"""Scene-level benchmark eval: 3 local models on benchmark_v1_scene_qa.
Resume-safe. Usage: python scripts/run_scene_eval.py [model_key ...]"""
import sys, json, re, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.cdw_vqa.models_api import LocalQwenVL

DATA = BASE / "data"
OUT = BASE / "results" / "scene_eval"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen3-vl-8b": dict(path=r"models/qwen3-vl-8b-instruct", max_tokens=64),
    "glm-4.1v-9b": dict(path=r"models/glm-4.1v-9b-thinking", max_tokens=1024),
    "qwen2.5-vl-7b": dict(path=r"models/qwen2.5-vl-7b-instruct", max_tokens=64),
}


def clean(a: str) -> str:
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
    a = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</?answer>", "", a)
    return a.strip()


CACHE = Path(r"datasets/plan3_scene_cache")


def cached_image(path: str, max_side: int = 1280) -> str:
    """Pre-resize huge images (CDW-Seg is 3000x4000) to keep vision tokens
    bounded; cached on disk. Scene-level QA does not need 12MP."""
    import cv2
    import numpy as np
    p = Path(path)
    if max(p.suffix.lower() == ".png", False):
        pass
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / (p.stem + ".jpg")
    if out.exists():
        return str(out)
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    buf.tofile(str(out))
    return str(out)


def prompt_of(it) -> str:
    q = it["question"]
    if it["qtype"] == "mcq":
        ch = " ".join(f"{k}. {v}" for k, v in it["choices"].items())
        return f"{q}\nChoices: {ch}\nAnswer with the letter only."
    return f"{q}\nAnswer Yes or No."


def score(it, raw: str) -> bool:
    a = clean(raw).lower()
    if it["qtype"] == "yesno":
        first = a.split()[0].strip(".,!") if a.split() else ""
        return first == it["answer"].lower()
    m = re.search(r"\b([abcd])\b", a)
    if m:
        return m.group(1).upper() == it["answer"]
    # fallback: answer text contained
    ans_text = it.get("answer_text") or it["choices"][it["answer"]]
    return ans_text.lower().replace("_", " ") in a


def main():
    keys = sys.argv[1:] or list(MODELS)
    qa = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
    print(f"items: {len(qa)}")
    for mk in keys:
        conf = MODELS[mk]
        resp_path = OUT / f"scene_{mk}.jsonl"
        done = set()
        if resp_path.exists():
            for l in open(resp_path, encoding="utf-8"):
                r = json.loads(l)
                done.add((r["dataset"], r["image_id"], r["family"], r["question"][:40]))
        todo = [it for it in qa
                if (it["dataset"], it["image_id"], it["family"], it["question"][:40]) not in done]
        print(f"\n=== {mk}: todo {len(todo)} (done {len(done)})")
        if not todo:
            continue
        model = LocalQwenVL(conf["path"])
        t0 = time.time()
        with open(resp_path, "a", encoding="utf-8") as f:
            for i, it in enumerate(todo):
                try:
                    raw = model.ask(prompt_of(it), cached_image(it["image_path"]),
                                    max_tokens=conf["max_tokens"])
                except Exception as e:
                    raw = f"ERROR {type(e).__name__}: {e}"
                rec = dict(it, raw=raw, correct=score(it, raw))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 300 == 0:
                    print(f"{mk} {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
