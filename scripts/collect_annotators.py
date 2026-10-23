"""Collect all annotator answers from the remote server and compute agreement.

Pulls answers/*.csv + logins.csv from the server, joins with the local truth
(human_baseline_v2_items.json), and reports:
  - per annotator: n answered, fine accuracy, PVC/metal recall, session times
  - pairwise Cohen's kappa (fine 3-way) between all annotators
  - Fleiss kappa across annotators (on commonly-answered items)
  - majority-vote accuracy (with tie handling reported)

Usage: python scripts/collect_annotators.py          # pull + analyse
       python scripts/collect_annotators.py --local  # analyse already-pulled files
"""
import csv, io, json, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
LOCAL_DIR = BASE / "data" / "annotators"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
import os
HOST = os.environ.get("LABEL_HOST", "")
USER = os.environ.get("LABEL_USER", "")
PW = os.environ.get("LABEL_PW", "")


def pull():
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PW, timeout=15)
    _, o, _ = c.exec_command("ls /root/deploy_cdw_label/answers/", timeout=15)
    files = [f for f in o.read().decode().split() if f.endswith(".csv")]
    s = c.open_sftp()
    for f in files:
        s.get(f"/root/deploy_cdw_label/answers/{f}", str(LOCAL_DIR / f))
    c.close()
    print(f"pulled {len(files)} files:", files)
    return files


def cohen_kappa(a, b):
    """a, b: lists of labels for the same items."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(table):
    """table: n_items x n_cats counts."""
    n_items = len(table)
    n_cats = len(table[0])
    n_raters = sum(table[0])
    if n_raters < 2:
        return float("nan")
    p = [sum(row[c] for row in table) / (n_items * n_raters) for c in range(n_cats)]
    pj = 0.0
    for row in table:
        pj += sum(x * x for x in row) - n_raters
    pj /= n_items * n_raters * (n_raters - 1)
    pe = sum(x * x for x in p)
    return (pj - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    if "--local" not in sys.argv:
        pull()
    items = {i["uid"]: i for i in
             json.load(open(BASE / "data" / "human_baseline_v2_items.json",
                            encoding="utf-8"))}
    ann = {}
    for f in sorted(LOCAL_DIR.glob("answers_*.csv")):
        name = f.stem[len("answers_"):]
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8-sig"))
                if r.get("uid") and r["uid"] != "__test__"]
        ann[name] = {r["uid"]: r["answer"] for r in rows}
    if not ann:
        print("no annotator files")
        return
    print("\n== per annotator ==")
    for name, a in ann.items():
        got = [(uid, ans) for uid, ans in a.items() if uid in items]
        n = len(got)
        if not n:
            print(f"{name}: 0 answered")
            continue
        ok = [ans == items[uid]["answer"] for uid, ans in got]
        pvc = [ (ans == items[uid]["answer"]) for uid, ans in got
               if items[uid]["answer"] == "PVC"]
        met = [ (ans == items[uid]["answer"]) for uid, ans in got
               if items[uid]["answer"] == "metal"]
        ts = [int(r["ts"]) for r in csv.DictReader(
            open(LOCAL_DIR / f"answers_{name}.csv", encoding="utf-8-sig"))
            if r.get("uid") and r["uid"] != "__test__" and r.get("ts")]
        span = f"{min(ts)}~{max(ts)}" if ts else "?"
        print(f"{name}: n={n}/200 acc={sum(ok)/n:.3f} "
              f"PVC-recall={sum(pvc)/max(1,len(pvc)):.3f}({len(pvc)}) "
              f"metal-recall={sum(met)/max(1,len(met)):.3f}({len(met)}) ts={span}")

    print("\n== pairwise Cohen's kappa (3-way) ==")
    for (n1, a1), (n2, a2) in combinations(ann.items(), 2):
        common = sorted(set(a1) & set(a2) & set(items))
        if common:
            k = cohen_kappa([a1[u] for u in common], [a2[u] for u in common])
            print(f"{n1} vs {n2}: k={k:.3f} (n={len(common)})")

    common = set(items)
    for a in ann.values():
        common &= set(a)
    common = sorted(common)
    if len(ann) >= 2 and common:
        cats = ["PVC", "metal", "other"]
        table = [[sum(1 for a in ann.values() if a[u] == c) for c in cats]
                 for u in common]
        print(f"\n== Fleiss kappa on {len(common)} fully-annotated items, "
              f"{len(ann)} annotators: {fleiss_kappa(table):.3f} ==")
        if len(ann) >= 2:
            correct = ties = 0
            for u in common:
                votes = [a[u] for a in ann.values()]
                top = max(set(votes), key=votes.count)
                if votes.count(top) > len(votes) / 2:
                    correct += (top == items[u]["answer"])
                else:
                    ties += 1
            print(f"majority vote (strict >50%, {ties} ties excluded): "
                  f"acc={correct / max(1, len(common) - ties):.3f}")


if __name__ == "__main__":
    main()
