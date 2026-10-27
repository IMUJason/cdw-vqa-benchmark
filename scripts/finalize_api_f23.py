"""Final rebuild of API F2/F3 rows: guarantee all 1,296 items present.

Keeps every usable response, re-queries anything missing / errored / empty
sequentially with a 3 s pause (the endpoint rate-limits bursts), and rewrites
the file incrementally so an interruption cannot lose work.

Usage: python scripts/finalize_api_f23.py [kimi|glm]
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from run_scene_eval import prompt_of, score as scene_score, cached_image
from run_api_eval import get_client, ask_retry

DATA = BASE / "data"
OUT = BASE / "results" / "api_eval"
KEYS = ("dataset", "image_id", "level", "split", "family", "qtype", "question",
        "choices", "answer", "answer_text", "image_path")


def k_of(r):
    return (r["dataset"], r.get("image_path", ""), r["family"], r["question"])


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "kimi"
    p = OUT / f"api_{name}_v2.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]

    qa = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
    f23 = [r for r in qa if r["family"] in ("F2_extent", "F3_dominance")]
    want = {k_of(r) for r in f23}

    carry, have = [], {}
    for r in rows:
        if r.get("level") == "scene" and r["family"] in ("F2_extent", "F3_dominance"):
            s = str(r.get("raw", "")).strip()
            if s and not s.startswith("ERROR"):
                have[k_of(r)] = r
        else:
            carry.append(r)

    todo = [it for it in f23 if k_of(it) not in have]
    print(f"{name}: carry {len(carry)}, usable F2/F3 {len(have)}, to query {len(todo)}", flush=True)

    def write():
        with open(p, "w", encoding="utf-8") as f:
            for r in carry:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in have.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    client = get_client(name)
    for i, it in enumerate(todo):
        time.sleep(3.0)
        raw = ask_retry(client, prompt_of(it), cached_image(it["image_path"]),
                        max_tokens=4096, tries=5)
        rec = dict({k: it[k] for k in KEYS if k in it}, raw=raw,
                   correct=scene_score(it, raw))
        s = str(raw).strip()
        if s and not s.startswith("ERROR"):
            have[k_of(it)] = rec
        else:
            print(f"  still failing: {it['family']} {it['dataset']} {it['image_id']} "
                  f"{s[:30]}", flush=True)
        if (i + 1) % 10 == 0:
            write()
            print(f"  {i+1}/{len(todo)} done, usable {len(have)}/{len(want)}", flush=True)
    write()

    ok = sum(1 for r in have.values() if str(r.get("raw", "")).strip())
    print(f"{name}: final F2/F3 usable {ok}/{len(want)}; file rows "
          f"{len(carry) + len(have)}")


if __name__ == "__main__":
    main()
