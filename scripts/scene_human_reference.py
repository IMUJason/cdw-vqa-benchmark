"""Human reference for the scene suite: reproduce the per-cell table from released data.

Inputs (both in this repository):
  data/benchmark_v1_scene_qa.jsonl        questions, options and answers
  data/annotators_scene/answers_*.csv     per-annotator responses (uid, answer, ts)

Output: per-annotator accuracy, per-cell mean with 95% CI, per-item majority vote,
Fleiss' kappa per family. No network, no server, no external state.

Usage: python scripts/scene_human_reference.py
"""
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
QA = BASE / "data" / "benchmark_v1_scene_qa.jsonl"
ANS = BASE / "data" / "annotators_scene"
FAMS = ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]
DSS = ["codd", "cdw_seg"]
MIN_ITEMS = 100


def load_answers():
    out = {}
    for p in sorted(ANS.glob("answers_*.csv")):
        who = p.stem[len("answers_"):]
        out[who] = {r["uid"]: r["answer"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
    return out


def fleiss(uids, ans, raters):
    n, k = len(uids), len(raters)
    cats = sorted({ans[w][u] for u in uids for w in raters})
    pbar = sum((sum(v * v for v in Counter(ans[w][u] for w in raters).values()) - k) / (k * (k - 1))
               for u in uids) / n
    pj = [sum(1 for u in uids for w in raters if ans[w][u] == c) / (n * k) for c in cats]
    pe = sum(x * x for x in pj)
    return (pbar - pe) / (1 - pe)


def main():
    rows = [json.loads(l) for l in open(QA, encoding="utf-8")]
    uid_truth, uid_fam, uid_ds = {}, {}, {}
    for r in rows:
        if r["family"] not in FAMS:
            continue
        uid = f"scene|{r['dataset']}|{r['image_id']}|{r['family']}"
        uid_truth[uid] = r["answer"]
        uid_fam[uid], uid_ds[uid] = r["family"], r["dataset"]

    ans = {w: a for w, a in load_answers().items() if len(a) >= MIN_ITEMS}
    print(f"annotators: {len(ans)} (>= {MIN_ITEMS}/120 items)")
    print(f"{'name':10s} {'n':>4s} {'accuracy':>9s}")
    for w in sorted(ans, key=lambda x: -len(ans[x])):
        sc = [(u, a) for u, a in ans[w].items() if u in uid_truth]
        acc = sum(1 for u, a in sc if a == uid_truth[u]) / len(sc)
        print(f"{w:10s} {len(ans[w]):4d} {acc:9.3f}")

    print("\nper-cell reference (mean over annotators, 95% CI; majority vote):")
    print(f"{'cell':20s} {'mean':>6s} {'CI':>15s} {'sd':>6s} {'majority':>9s} {'raters':>6s}")
    for fam in FAMS:
        for ds in DSS:
            uids = [u for u in uid_truth if uid_fam[u] == fam and uid_ds[u] == ds
                    and all(u in ans[w] for w in ans)]
            per = [sum(1 for u in uids if ans[w][u] == uid_truth[u]) / len(uids) for w in ans]
            m, sd = statistics.mean(per), statistics.pstdev(per)
            ci = 1.96 * sd / (len(per) ** 0.5)
            def majority(u):
                (win, cnt), *rest = Counter(ans[w][u] for w in ans).most_common()
                return win if (not rest or rest[0][1] < cnt) else None
            maj = sum(1 for u in uids if majority(u) == uid_truth[u]) / len(uids)
            print(f"{fam[:2] + ' ' + ds:20s} {m:6.3f} [{m - ci:6.3f},{m + ci:6.3f}] "
                  f"{sd:6.3f} {maj:9.3f} {len(per):6d}")

    print("\nFleiss' kappa per family:")
    for fam in FAMS:
        uids = [u for u in uid_truth if uid_fam[u] == fam and all(u in ans[w] for w in ans)]
        print(f"  {fam:18s} kappa={fleiss(uids, ans, list(ans)):+.3f}  "
              f"items={len(uids)} raters={len(ans)}")
    return uid_truth, uid_fam


def matched_models(uid_truth, uid_fam):
    """Same 120 items, model side, unified answer rule (rescore_scene)."""
    sys.path.insert(0, str(BASE / "scripts"))
    from rescore_scene import unified, hit
    files = {"Qwen3-VL": "scene_qwen3-vl-8b.jsonl", "Qwen2.5-VL": "scene_qwen2.5-vl-7b.jsonl",
             "GLM-4.1V": "scene_glm-4.1v-9b.jsonl",
             "Kimi-k3": "../api_eval/api_kimi_v2.jsonl", "GLM-5.3": "../api_eval/api_glm_v2.jsonl"}
    want = {(uid.split("|")[1], uid.split("|")[2], uid_fam[uid]) for uid in uid_truth}
    print("\nmatched-item model accuracy (same 120 items, unified rule):")
    hdr = "model        "
    print(hdr + "".join(f"{f[:2]+' '+d[:3]:>12s}" for f in FAMS for d in DSS))
    for name, fn in files.items():
        p = BASE / "results" / "scene_eval" / fn
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if fn.startswith("../"):
            rows = [r for r in rows if r.get("level") == "scene"]
        cells = {}
        for fam in FAMS:
            for ds in DSS:
                sel = [r for r in rows if r["family"] == fam and r.get("dataset") == ds
                       and (ds, str(r.get("image_id")), fam) in want]
                if sel:
                    cells[(fam, ds)] = sum(hit(r, unified(r)) for r in sel) / len(sel)
        print(f"{name:12s}" + "".join(f"{cells.get((f, d), float('nan')):12.3f}"
                                      for f in FAMS for d in DSS))


if __name__ == "__main__":
    _truth, _fam = main()
    matched_models(_truth, _fam)
