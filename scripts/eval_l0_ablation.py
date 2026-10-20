"""Instance-suite-only evaluation of an adapter (used for the L0-only
ablation): test split, all four arms, object + material questions.
Usage: python scripts/eval_l0_ablation.py <key>
Env: BASE_M, ADAPTER_M to override CONFIGS.
"""
import sys, json, os, re, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from fine_metrics import obj_score, mat_score, clean  # symmetric two-level scorer

DATA = BASE / "data"
OUT = BASE / "results" / "adapted_l0"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = Path(r"datasets/plan3_arms")

CONFIGS = {
    "qwen3vl_l0": dict(base=r"models/qwen3-vl-8b-instruct",
                       adapter=r"models/adapters/qwen3vl-cdw-lora-r16-l0",
                       max_tokens=64),
    "qwen25vl_l0": dict(base=r"models/qwen2.5-vl-7b-instruct",
                        adapter=r"models/adapters/qwen25vl-cdw-lora-r16-l0",
                        max_tokens=64),
}


def main():
    key = sys.argv[1]
    conf = CONFIGS[key]
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    model = AutoModelForImageTextToText.from_pretrained(
        conf["base"], dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, conf["adapter"])
    model.eval()
    processor = AutoProcessor.from_pretrained(conf["base"])

    def ask(q, img):
        msgs = [{"role": "user", "content": [{"type": "image", "image": str(img)},
                                             {"type": "text", "text": q}]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        imgs, _ = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=conf["max_tokens"])
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()

    inst = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
    inst = [it for it in inst if not it.get("excluded") and not it.get("needs_manual_label")
            and it["split"] == "test" and it.get("cls") == "pipes"]
    print(f"instance test pipes items: {len(inst)}")

    resp_path = OUT / f"l0ab_{key}.jsonl"
    done = set()
    if resp_path.exists():
        for l in open(resp_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["diag_id"], r["arm"], r["family"]))
    t0 = time.time()
    n = 0
    with open(resp_path, "a", encoding="utf-8") as f:
        for it in inst:
            if (it["diag_id"], it["arm"], it["family"]) in done:
                continue
            q = it["question"] + " Answer with one word or a short phrase."
            raw = ask(q, ARMS / it["image"])
            f.write(json.dumps(dict(it, raw=raw), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"{n} {time.time()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(resp_path, encoding="utf-8")]
    print("\n====", key, "(L0-only trained, test split) ====")
    for fam in ["instance_object", "instance_material"]:
        for arm in ["L0", "L1", "L2", "L3"]:
            sel = [r for r in rows if r["family"] == fam and r["arm"] == arm]
            if fam == "instance_object":
                acc = sum(obj_score(r["raw"]) for r in sel) / len(sel)
                print(f"{fam:18s} {arm} n={len(sel)} obj={acc:.3f}")
            else:
                fine = sum(mat_score(r["raw"], r["answer"])[0] for r in sel) / len(sel)
                coarse = sum(mat_score(r["raw"], r["answer"])[1] for r in sel) / len(sel)
                print(f"{fam:18s} {arm} n={len(sel)} mat_fine={fine:.3f} mat_coarse={coarse:.3f}")
    print("DONE", key)


if __name__ == "__main__":
    main()
