"""Re-evaluate GLM-4.1V-9B-Thinking on the rebuilt-key scene F2/F3 items.

Why: `results/scene_eval/scene_glm-4.1v-9b.jsonl` (Oct 12) was produced before the
answer-key rebuild; its MCQ keys are A-biased (a=531/1296) and its choice sets
differ from the released corpus, so those answers cannot be scored against the
rebuilt items. This script runs only F2_extent and F3_dominance (n=1,296) with the
protocol of the paper (GLM budget 1,024 new tokens, 1,280-px image cache).

Resume-safe; writes results/scene_eval/scene_glm-4.1v-9b_f23v2.jsonl.
Usage: python scripts/rerun_scene_glm_f23.py
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

from run_scene_eval import MODELS, cached_image, prompt_of, score  # noqa: E402
from src.cdw_vqa.models_api import LocalQwenVL  # noqa: E402

QA = BASE / "data" / "benchmark_v1_scene_qa.jsonl"
OUT = BASE / "results" / "scene_eval" / "scene_glm-4.1v-9b_f23v2.jsonl"
KEYS = {"F2_extent", "F3_dominance"}


def key_of(r):
    return (r["dataset"], r["image_id"], r["family"], json.dumps(r.get("choices"), sort_keys=True))


def main():
    qa = [json.loads(l) for l in open(QA, encoding="utf-8")]
    todo_all = [it for it in qa if it["family"] in KEYS]
    done = set()
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            done.add(key_of(r))
    todo = [it for it in todo_all if key_of(it) not in done]
    print(f"items F2/F3: {len(todo_all)}  done: {len(done)}  todo: {len(todo)}", flush=True)
    if not todo:
        print("ALL DONE (nothing to do)")
        return

    conf = MODELS["glm-4.1v-9b"]
    model = LocalQwenVL(conf["path"])
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            try:
                raw = model.ask(prompt_of(it), cached_image(it["image_path"]),
                                max_tokens=conf["max_tokens"])
            except Exception as e:  # keep going; record the failure
                raw = f"ERROR {type(e).__name__}: {e}"
            rec = dict(it, raw=raw, correct=score(it, raw))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == 1:
                el = time.time() - t0
                print(f"{i}/{len(todo)}  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min",
                      flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
