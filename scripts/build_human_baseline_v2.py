"""Human baseline v2: balanced, fine-grained, blind.

v1 (data/human_baseline.csv) was 80% PVC and free-text: the annotator answered
"塑料" to all 80 instance questions, so 0.80 equalled the majority-class rate
and metal recall was 0/16. v2 fixes this:
  * 25 PVC + 25 metal pipes instances, drawn from the TEST split so the
    numbers are directly comparable with model test-split numbers;
  * forced three-way choice (PVC/塑料 | 金属 | 其他) -- fine-grained, no
    free text, no coarse "plastic" escape;
  * all four arms per instance (50 x 4 = 200 questions);
  * annotator blind to truth (no outline shown, arm image only).
"""
import csv, json, random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
random.seed(20261019)

truth = {r["diag_id"]: r["material"] for r in csv.DictReader(
    open(DATA / "pipes_material_final.csv", encoding="utf-8-sig"))}
qa = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
mat_q = [it for it in qa if it["family"] == "instance_material"
         and it["cls"] == "pipes" and it["split"] == "test"
         and not it.get("excluded") and not it.get("needs_manual_label")]
mat_q = [it for it in mat_q if str(it["diag_id"]) in truth]

ids_pvc = sorted({str(it["diag_id"]) for it in mat_q if truth[str(it["diag_id"])] == "PVC"})
ids_met = sorted({str(it["diag_id"]) for it in mat_q if truth[str(it["diag_id"])] == "metal"})
print(f"test pipes: PVC {len(ids_pvc)}, metal {len(ids_met)}")
sel_pvc = random.sample(ids_pvc, 25)
sel_met = ids_met[:25] if len(ids_met) >= 25 else ids_met
sel = set(sel_pvc) | set(sel_met)

items = []
for it in mat_q:
    if str(it["diag_id"]) in sel:
        items.append(dict(diag_id=it["diag_id"], arm=it["arm"],
                          family="instance_material", qtype="pvc3",
                          question=it["question"], answer=truth[str(it["diag_id"])],
                          image=it["image"], split="test",
                          uid=f"v2|{it['diag_id']}|{it['arm']}"))
random.shuffle(items)
with open(DATA / "human_baseline_v2_items.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=1)
from collections import Counter
print("items:", len(items), Counter(i["arm"] for i in items),
      Counter(i["answer"] for i in items))
