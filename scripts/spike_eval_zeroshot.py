"""Spike: zero-shot local Qwen2.5-VL-7B on spike arm images (L0 vs L1 pairing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdw_vqa.models_api import LocalQwenVL, KimiVision

ARMS = Path(__file__).resolve().parents[1] / "results" / "spike_arms"
OUT = Path(__file__).resolve().parents[1] / "results" / "spike_eval"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    (ARMS / "s4_concrete_L0.jpg", "What material dominates the visible waste in this image? Answer with one word."),
    (ARMS / "s4_concrete_L1.jpg", "What material is the isolated fragment mainly made of? Answer with one word."),
    (ARMS / "s0_pipes_L0.jpg", "What type of waste objects are visible on the belt? Answer with one word."),
    (ARMS / "s0_pipes_L1.jpg", "What material is the isolated fragment mainly made of? Answer with one word."),
]

def main():
    qwen = LocalQwenVL()
    kimi = KimiVision("k3")
    lines = []
    for img, q in CASES:
        if not img.exists():
            print("MISSING", img); continue
        a_local = qwen.ask(q, img)
        a_kimi = kimi.ask(q, img)
        lines.append(f"IMG {img.name}\n Q: {q}\n local7B: {a_local}\n kimi_k3: {a_kimi}\n")
        print(lines[-1])
    (OUT / "zeroshot_compare.txt").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", OUT)

if __name__ == "__main__":
    main()
