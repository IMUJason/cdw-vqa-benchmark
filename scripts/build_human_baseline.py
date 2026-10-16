"""Build human-baseline item list: 120 scene (F1-F4 x 2 datasets) + 80 instance
(20 pipes x 4 arms, material-Q). Writes data/human_baseline_items.json."""
import json, random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
random.seed(20261012)

scene = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
inst = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
inst = [it for it in inst if not it.get("excluded") and not it.get("needs_manual_label")]

items = []
for fam in ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]:
    for ds in ["codd", "cdw_seg"]:
        pool = [it for it in scene if it["family"] == fam and it["dataset"] == ds]
        for it in random.sample(pool, min(15, len(pool))):
            it2 = dict(it)
            it2["uid"] = f"scene|{ds}|{it['image_id']}|{fam}|{it['question'][:30]}"
            items.append(it2)

man = {d["diag_id"]: d for d in json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]}
pipes_ids = [did for did, d in man.items() if d["cls"] == "pipes"]
random.shuffle(pipes_ids)
for did in pipes_ids[:20]:
    for r in inst:
        if r["diag_id"] == did and r["family"] == "instance_material":
            it2 = dict(r)
            it2["uid"] = f"inst|{did}|{r['arm']}"
            items.append(it2)

json.dump(items, open(DATA / "human_baseline_items.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
from collections import Counter
print("total:", len(items), Counter(i["family"] for i in items))
