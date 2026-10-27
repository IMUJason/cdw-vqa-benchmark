"""Re-run zero-shot scene eval for the rebuilt F2/F3 keys (position-balanced).

Keeps F1/F4 responses verbatim from the archived v1 files (those items are
byte-identical), re-queries F2/F3 only (1,296 items per model), merges into the
canonical results/scene_eval/scene_<model>.jsonl.

Usage: python scripts/rerun_scene_f23.py [model_key ...]
"""
import sys, json, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from run_scene_eval import MODELS, prompt_of, score, cached_image
from src.cdw_vqa.models_api import LocalQwenVL

DATA = BASE / "data"
OUT = BASE / "results" / "scene_eval"

qa = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
f23 = {(r["dataset"], r["image_id"], r["family"], r["question"]): r
       for r in qa if r["family"] in ("F2_extent", "F3_dominance")}

keys = sys.argv[1:] or list(MODELS)
for mk in keys:
    conf = MODELS[mk]
    arch = OUT / f"scene_{mk}_v1biased.jsonl"
    cur = OUT / f"scene_{mk}.jsonl"
    rows = [json.loads(l) for l in open(arch, encoding="utf-8")]
    keep, todo = [], []
    for r in rows:
        k = (r["dataset"], r["image_id"], r["family"], r["question"])
        if r["family"] in ("F1_presence", "F4_contamination"):
            keep.append(r)
        else:
            item = f23.get(k)
            assert item is not None, k
            todo.append(item)
    print(f"\n=== {mk}: keep F1/F4 {len(keep)}, re-run F2/F3 {len(todo)}")
    model = LocalQwenVL(conf["path"])
    t0 = time.time()
    done = 0
    with open(cur, "w", encoding="utf-8") as f:
        for it in todo:
            try:
                raw = model.ask(prompt_of(it), cached_image(it["image_path"]),
                                max_tokens=conf["max_tokens"])
            except Exception as e:
                raw = f"ERROR {type(e).__name__}: {e}"
            f.write(json.dumps(dict(it, raw=raw, correct=score(it, raw)),
                               ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 200 == 0:
                print(f"{mk} {done}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    del model
    import torch
    torch.cuda.empty_cache()
    print(f"{mk} DONE ({time.time()-t0:.0f}s)")
print("ALL DONE")
