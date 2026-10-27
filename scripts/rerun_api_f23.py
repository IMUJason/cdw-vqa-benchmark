"""Concurrent re-eval of API models on the rebuilt F2/F3 keys.

Instance rows and F1/F4 responses are carried over verbatim from the archived
v1 files; the full F2/F3 subset (1,296 items per model) is re-queried and merged
into results/api_eval/api_<name>_v2.jsonl.

Kimi-k3 is a reasoning model, so both API models run with a 2,048-token budget;
responses that terminate without content are kept in the file and handled by
the unified delivery policy (success rate / end-to-end / conditional accuracy).

Concurrency 6 keeps wall-clock near two hours for 1,296 items.

Usage: python scripts/rerun_api_f23.py [kimi|glm ...] [workers]
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

DATA = BASE / "data"
OUT = BASE / "results" / "api_eval"
MAX_TOKENS = 2048

qa = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
f23 = [r for r in qa if r["family"] in ("F2_extent", "F3_dominance")]


def main():
    names = [a for a in sys.argv[1:] if not a.isdigit()] or ["kimi", "glm"]
    workers = next((int(a) for a in sys.argv[1:] if a.isdigit()), 6)
    for name in names:
        cur = OUT / f"api_{name}_v2.jsonl"
        rows = [json.loads(l) for l in open(cur, encoding="utf-8")]
        seen, carry = set(), []
        for r in rows:
            if r.get("level") == "scene" and r["family"] in ("F2_extent", "F3_dominance"):
                continue
            k = r.get("uid") or json.dumps(r, sort_keys=True)[:100]
            if k in seen:
                continue
            seen.add(k)
            carry.append(r)
        with open(cur, "w", encoding="utf-8") as f:
            for r in carry:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n=== {name}: carry {len(carry)}, re-running {len(f23)} F2/F3 "
              f"at {MAX_TOKENS} tokens, {workers} workers", flush=True)

        client = get_client(name)
        done = 0
        t0 = time.time()

        def work(it):
            raw = ask_retry(client, prompt_of(it), cached_image(it["image_path"]),
                            max_tokens=MAX_TOKENS)
            return dict(it, raw=raw, correct=scene_score(it, raw))

        with open(cur, "a", encoding="utf-8") as f, \
                ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, it) for it in f23]
            for fut in as_completed(futs):
                r = fut.result()
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 100 == 0:
                    el = time.time() - t0
                    print(f"{name} {done}/{len(f23)} {el:.0f}s "
                          f"({el / done:.1f}s/item)", flush=True)
        with open(cur, "a", encoding="utf-8") as f:
            for r in carry:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name} DONE {done} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
