"""Score human baseline v2 (balanced, fine-grained, forced choice).
Fine scoring: PVC button -> PVC, metal button -> metal, other -> wrong.
Reports per-arm accuracy, PVC/metal recall, cross-arm consistency."""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

items = json.load(open(DATA / "human_baseline_v2_items.json", encoding="utf-8"))
answers = {r["uid"]: r["answer"] for r in csv.DictReader(
    open(DATA / "human_baseline_v2.csv", encoding="utf-8-sig")) if r["uid"] != "__test__"}

acc = defaultdict(lambda: [0, 0])
rec = defaultdict(lambda: [0, 0])
byid = defaultdict(dict)
unanswered = 0
for it in items:
    a = answers.get(it["uid"])
    if a is None:
        unanswered += 1
        continue
    ok = (a == it["answer"])
    acc[it["arm"]][0] += int(ok)
    acc[it["arm"]][1] += 1
    rec[(it["arm"], it["answer"])][0] += int(ok)
    rec[(it["arm"], it["answer"])][1] += 1
    byid[str(it["diag_id"])][it["arm"]] = a

print(f"items {len(items)}, unanswered {unanswered}")
for arm in ["L0", "L1", "L2", "L3"]:
    c, n = acc[arm]
    pvc_c, pvc_n = rec[(arm, "PVC")]
    met_c, met_n = rec[(arm, "metal")]
    print(f"{arm}: acc {c}/{n}={c/n:.3f}   PVC recall {pvc_c}/{pvc_n}={pvc_c/pvc_n:.3f}"
          f"   metal recall {met_c}/{met_n}={met_c/met_n:.3f}")
same = sum(1 for d, arms in byid.items() if len(set(arms.values())) == 1)
print(f"cross-arm consistency: {same}/{len(byid)} = {same/len(byid):.3f}")
print("answer distribution:", Counter(a for a in answers.values() if a != "__test__"))
