"""Prefill pipes material labels with local Qwen3-VL on L3 isolated crops.
Writes pipes_prefill.csv (diag_id, prefill, raw_answer) for human verification."""
import sys, csv, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.models_api import LocalQwenVL
import json

DATA = Path(__file__).resolve().parents[1] / "data"
ARMS = Path(r"datasets/plan3_arms")

Q = ("This image shows one isolated waste object on a plain background. "
     "Is it made of metal, plastic (PVC), or something else? "
     "Answer with exactly one word: metal, plastic, or other.")

def main():
    man = json.load(open(DATA / "diagnostic_v2_manifest.json", encoding="utf-8"))["instances"]
    pipes = [d for d in man if d["cls"] == "pipes"]
    m = LocalQwenVL(r"models/qwen3-vl-8b-instruct")
    rows = []
    for i, d in enumerate(pipes):
        img = ARMS / f"{d['diag_id']:03d}_L3.jpg"
        a = m.ask(Q, img, max_tokens=64).strip().lower()
        lab = "metal" if "metal" in a else ("PVC" if ("plastic" in a or "pvc" in a) else "other")
        rows.append((d["diag_id"], img.name, lab, a[:60]))
        if i % 25 == 0:
            print(i, d["diag_id"], lab, a[:40], flush=True)
    with open(DATA / "pipes_prefill.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diag_id", "image_L3", "prefill(metal/PVC/other)", "raw_answer"])
        w.writerows(rows)
    print("DONE", len(rows))

if __name__ == "__main__":
    main()
