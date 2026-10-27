"""Refill empty/errored instance rows for an API results file (all classes).

Pipes use the symmetric two-level scorer; other classes use the original
class-word scorer. Writes incrementally so an interruption keeps progress.

Usage: python scripts/refill_api_inst.py kimi
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from fine_metrics import obj_score, mat_score
from run_w6_gate import score as w6_score
from run_api_eval import get_client, ask_retry

OUT = BASE / "results" / "api_eval"
ARMS = Path(r"datasets/plan3_arms")


def usable(raw):
    s = str(raw).strip()
    return bool(s) and not s.startswith("ERROR")


def score_row(r, raw):
    if r.get("cls") != "pipes":
        return bool(w6_score(r, raw))
    if r["family"] == "instance_object":
        return bool(obj_score(raw))
    return bool(mat_score(raw, r["answer"])[1])


def main():
    name = sys.argv[1]
    p = OUT / f"api_{name}_v2.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    todo = [r for r in rows if r.get("level") != "scene" and not usable(r.get("raw", ""))]
    print(f"{name}: instance rows {sum(1 for r in rows if r.get('level')!='scene')}, "
          f"to refill {len(todo)}", flush=True)
    client = get_client(name)

    def work(r):
        q = r["question"] + " Answer with one word or a short phrase."
        raw = ask_retry(client, q, ARMS / r["image"], max_tokens=2048, tries=5)
        return r, raw

    def dump():
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fixed = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            r, raw = fut.result()
            if usable(raw):
                r["raw"] = raw
                try:
                    r["correct"] = score_row(r, raw)
                except Exception:
                    r["correct"] = False
                fixed += 1
            if fixed and fixed % 50 == 0:
                dump()
                print(f"  {fixed}/{len(todo)}", flush=True)
            time.sleep(0.3)
    dump()
    left = sum(1 for r in rows if r.get("level") != "scene" and not usable(r.get("raw", "")))
    print(f"{name}: repaired {fixed}/{len(todo)}; still missing {left}", flush=True)


if __name__ == "__main__":
    main()
