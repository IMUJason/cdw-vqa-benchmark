"""Scene-test + instance-L1-test subset evaluation for the v2-retrained
GLM-4.1V (full 2,407-item eval is ~13.5 h; the quoted GLM numbers live in the
scene test cells and the isolated-arm floor claims).

Usage: python scripts/eval_adapted_glm_subset.py
"""
import sys, json, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from run_scene_eval import prompt_of, score as scene_score, cached_image
from fine_metrics import obj_score, mat_score

DATA = BASE / "data"
OUT = BASE / "results" / "adapted"
ARMS = Path(r"datasets/plan3_arms")
CONF = dict(base=r"models/glm-4.1v-9b-thinking",
            adapter=r"models/adapters/glm41v-cdw-lora-r16-v2", max_tokens=1024)
SPLIT = {"test": "test", "testing": "test"}

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
model = AutoModelForImageTextToText.from_pretrained(CONF["base"], dtype=torch.bfloat16,
                                                    device_map="cuda")
model = PeftModel.from_pretrained(model, CONF["adapter"])
model.eval()
processor = AutoProcessor.from_pretrained(CONF["base"])


def ask(q, img):
    msgs = [{"role": "user", "content": [{"type": "image", "image": str(img)},
                                         {"type": "text", "text": q}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    imgs, _ = process_vision_info(msgs)
    inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONF["max_tokens"])
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()


resp = OUT / "adapted_glm41v_v2.jsonl"
done = set()
if resp.exists():
    for l in open(resp, encoding="utf-8"):
        r = json.loads(l)
        done.add((r.get("family"), str(r.get("image_id", r.get("diag_id"))),
                  r.get("arm", "scene"), r["question"][:40]))

scene = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
scene = [it for it in scene if SPLIT.get(it["split"]) == "test"]
inst = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
inst = [it for it in inst if not it.get("excluded") and not it.get("needs_manual_label")
        and it["split"] == "test" and it.get("cls") == "pipes" and it["arm"] == "L1"]
todo = scene + inst
print("items:", len(todo), f"({len(scene)} scene test + {len(inst)} inst L1 test)")

t0 = time.time()
n = 0
with open(resp, "a", encoding="utf-8") as f:
    for it in todo:
        k = (it["family"], str(it.get("image_id", it.get("diag_id"))),
             it.get("arm", "scene"), it["question"][:40])
        if k in done:
            continue
        if it.get("level") == "scene" or "image_path" in it:
            raw = ask(prompt_of(it), cached_image(it["image_path"]))
            ok = scene_score(it, raw)
        else:
            q = it["question"] + " Answer with one word or a short phrase."
            raw = ask(q, ARMS / it["image"])
            ok = (obj_score(raw) if it["family"] == "instance_object"
                  else mat_score(raw, it["answer"])[1])
        f.write(json.dumps(dict(it, raw=raw, correct=bool(ok)), ensure_ascii=False) + "\n")
        f.flush()
        n += 1
        if n % 50 == 0:
            print(f"{n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
print("DONE", n)
