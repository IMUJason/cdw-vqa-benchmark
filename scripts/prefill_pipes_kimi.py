"""Third judge: Kimi k3 votes only on disputed pipes ids (to_judge list)."""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.models_api import KimiVision

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
    todo = [r["diag_id"] for r in csv.DictReader(open(DATA / "pipes_to_judge.csv", encoding="utf-8-sig"))]
    m = KimiVision("k3")
    rows = []
    for i, did in enumerate(todo):
        img = ARMS / f"{int(did):03d}_L3.jpg"
        try:
            a = m.ask(Q, img, max_tokens=1024)
        except Exception as e:
            a = f"ERROR {type(e).__name__}"
        rows.append((did, img.name, label_of(a), a[:60].replace("\n", " ")))
        if i % 20 == 0:
            print(i, did, rows[-1][2], rows[-1][3][:40], flush=True)
    with open(DATA / "pipes_prefill_kimi.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diag_id", "image_L3", "kimi_prefill", "kimi_raw"])
        w.writerows(rows)
    print("DONE", len(rows))

if __name__ == "__main__":
    main()
