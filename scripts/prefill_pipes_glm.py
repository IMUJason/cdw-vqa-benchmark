"""Second-pass pipes material labeling with local GLM-4.1V-9B (dual-judge protocol).
Writes pipes_prefill_glm.csv; then merge_judges.py builds auto_accept/to_judge."""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.models_api import LocalQwenVL
import json, re

DATA = Path(__file__).resolve().parents[1] / "data"
ARMS = Path(r"datasets/plan3_arms")

Q = ("This image shows one isolated waste object on a plain background. "
     "Is it made of metal, plastic (PVC), or something else? "
     "Answer with exactly one word: metal, plastic, or other.")

def label_of(a: str) -> str:
    a = a.lower()
    if "metal" in a:
        return "metal"
    if "plastic" in a or "pvc" in a:
        return "PVC"
    return "other"

def main():
    man = json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]
    pipes = [d for d in man if d["cls"] == "pipes"]
    m = LocalQwenVL(r"models/glm-4.1v-9b-thinking")
    rows = []
    for i, d in enumerate(pipes):
        img = ARMS / f"{d['diag_id']:03d}_L3.jpg"
        a = m.ask(Q, img, max_tokens=512)
        a = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
        a = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>", "", a).strip()
        lab = label_of(a)
        rows.append((d["diag_id"], img.name, lab, a[:60].replace("\n", " ")))
        if i % 25 == 0:
            print(i, d["diag_id"], lab, a[:40], flush=True)
    with open(DATA / "pipes_prefill_glm.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diag_id", "image_L3", "glm_prefill", "glm_raw"])
        w.writerows(rows)
    print("DONE", len(rows))

if __name__ == "__main__":
    main()
