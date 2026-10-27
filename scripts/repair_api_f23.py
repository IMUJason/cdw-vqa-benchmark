"""Repair pass for API F2/F3 rows: retry rate-limit errors and empty
(non-terminating) responses sequentially with backoff.

Usage: python scripts/repair_api_f23.py [kimi|glm]
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

OUT = BASE / "results" / "api_eval"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "kimi"
    p = OUT / f"api_{name}_v2.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    good, rate_limited, empties = [], [], []
    for r in rows:
        if not (r.get("level") == "scene" and r["family"] in ("F2_extent", "F3_dominance")):
            good.append(r)
            continue
        s = str(r.get("raw", "")).strip()
        if s.startswith("ERROR"):
            rate_limited.append(r)
        elif not s:
            empties.append(r)
        else:
            good.append(r)
    print(f"{name}: good {len(good)}, rate-limited {len(rate_limited)}, empty {len(empties)}")

    client = get_client(name)

    def retry(batch, budget, label):
        fixed, still = 0, []
        for i, r in enumerate(batch):
            it = {k: r[k] for k in ("dataset", "image_id", "level", "split", "family",
                                    "qtype", "question", "choices", "answer", "answer_text",
                                    "image_path") if k in r}
            time.sleep(0.8)
            raw = ask_retry(client, prompt_of(it), cached_image(it["image_path"]),
                            max_tokens=budget, tries=4)
            rr = dict(it, raw=raw, correct=scene_score(it, raw))
            if str(raw).strip() and not str(raw).startswith("ERROR"):
                good.append(rr)
                fixed += 1
            else:
                still.append(rr)
            if (i + 1) % 20 == 0:
                print(f"  {label} {i+1}/{len(batch)} fixed {fixed}", flush=True)
        return fixed, still

    f1, s1 = retry(rate_limited, 2048, "rate-limit")
    f2, s2 = retry(empties, 4096, "empty->4096")
    good.extend(s2)
    with open(p, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: repaired {f1} rate-limited, {f2}/{len(empties)} empties; "
          f"remaining non-answered {len(s1)+len(s2)}; file {len(good)} rows")


if __name__ == "__main__":
    main()
