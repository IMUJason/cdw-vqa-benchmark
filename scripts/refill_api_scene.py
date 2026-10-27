"""Refill empty/errored scene rows in an API results file.

Only items without a usable answer are re-queried (2048 tokens, 4 workers);
the file is rewritten in place with the repaired rows.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from run_scene_eval import prompt_of, score as scene_score, cached_image
from run_api_eval import get_client, ask_retry

OUT = BASE / "results" / "api_eval"
KEYS = ("dataset", "image_id", "level", "split", "family", "qtype", "question",
        "choices", "answer", "answer_text", "image_path", "uid")


def usable(r):
    s = str(r.get("raw", "")).strip()
    return bool(s) and not s.startswith("ERROR")


def main():
    name = sys.argv[1]
    p = OUT / f"api_{name}_v2.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    todo = [r for r in rows if r.get("level") == "scene" and not usable(r)]
    print(f"{name}: scene rows {sum(1 for r in rows if r.get('level')=='scene')}, to refill {len(todo)}")
    client = get_client(name)

    def work(r):
        it = {k: r[k] for k in KEYS if k in r}
        raw = ask_retry(client, prompt_of(it), cached_image(it["image_path"]),
                        max_tokens=2048, tries=5)
        return r, raw

    fixed = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            r, raw = fut.result()
            if usable({"raw": raw}):
                r["raw"] = raw
                r["correct"] = scene_score(
                    {k: r[k] for k in KEYS if k in r}, raw)
                fixed += 1
            time.sleep(0.5)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    left = sum(1 for r in rows if r.get("level") == "scene" and not usable(r))
    print(f"{name}: repaired {fixed}/{len(todo)}; still missing {left}")


if __name__ == "__main__":
    main()
