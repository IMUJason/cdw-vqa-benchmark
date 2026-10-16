"""API frontier evaluation: Kimi k3 + GLM-5.3 (+ GPT relay on a subset).
Runs scene suite + instance suite with identical prompts/scoring as local eval.
Resume-safe. Usage: python scripts/run_api_eval.py [kimi|glm|gpt]"""
import sys, json, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from run_scene_eval import prompt_of, score as scene_score, cached_image
from run_w6_gate import score as inst_score
from src.cdw_vqa.models_api import KimiVision, AnthropicVision
from openai import OpenAI

DATA = BASE / "data"
ARMS = Path(r"datasets/plan3_arms")
OUT = BASE / "results" / "api_eval"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test",
             "train": "train", "val": "val", "test": "test"}


def load_items():
    scene = [json.loads(l) for l in open(DATA / "benchmark_v1_scene_qa.jsonl", encoding="utf-8")]
    inst = [json.loads(l) for l in open(DATA / "diagnostic_v2_qa.jsonl", encoding="utf-8")]
    inst = [it for it in inst if not it.get("excluded") and not it.get("needs_manual_label")]
    return scene, inst


def get_client(which):
    if which == "kimi":
        return KimiVision("k3")
    if which == "glm":
        return AnthropicVision("glm")
    kv = dict(l.strip().split("=", 1) for l in
              open(r"secrets.env", encoding="utf-8").read().splitlines() if "=" in l)

    class GPT:
        def __init__(self):
            self.c = OpenAI(api_key=kv["GPT_API_KEY"], base_url=kv["GPT_BASE_URL"] + "/v1")

        def ask(self, q, img, max_tokens=512):
            from src.cdw_vqa.models_api import img_data_url
            r = self.c.chat.completions.create(
                model=kv["GPT_MODEL"], max_tokens=max_tokens,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": img_data_url(img)}},
                    {"type": "text", "text": q}]}])
            return (r.choices[0].message.content or "").strip()
    return GPT()


def ask_retry(client, q, img, max_tokens, tries=3):
    for t in range(tries):
        try:
            return client.ask(q, img, max_tokens=max_tokens)
        except Exception as e:
            if t == tries - 1:
                return f"ERROR {type(e).__name__}"
            time.sleep(3 * (t + 1))


def main():
    which = sys.argv[1]
    scene, inst = load_items()
    if which == "gpt":  # 中转贵：只跑分层子集
        random_state = __import__("random").Random(20261012)
        scene = random_state.sample(scene, 200)
        inst = random_state.sample(inst, 200)
    resp_path = OUT / f"api_{which}.jsonl"
    done = set()
    if resp_path.exists():
        for l in open(resp_path, encoding="utf-8"):
            r = json.loads(l)
            done.add(r["uid"])
    client = get_client(which)
    n = 0
    t0 = time.time()
    with open(resp_path, "a", encoding="utf-8") as f:
        for it in scene:
            uid = f"scene|{it['dataset']}|{it['image_id']}|{it['family']}|{it['question'][:40]}"
            if uid in done:
                continue
            q = prompt_of(it)
            raw = ask_retry(client, q, cached_image(it["image_path"]),
                            1024 if which == "glm" else 512)
            f.write(json.dumps(dict(it, uid=uid, raw=raw, correct=scene_score(it, raw)),
                               ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"{which} scene {n} {time.time() - t0:.0f}s", flush=True)
        for it in inst:
            uid = f"inst|{it['diag_id']}|{it['arm']}|{it['family']}"
            if uid in done:
                continue
            q = it["question"] + " Answer with one word or a short phrase."
            raw = ask_retry(client, q, ARMS / it["image"],
                            1024 if which == "glm" else 512)
            f.write(json.dumps(dict(it, uid=uid, raw=raw, correct=inst_score(it, raw)),
                               ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"{which} inst {n} {time.time() - t0:.0f}s", flush=True)
    print("DONE", which, n)


if __name__ == "__main__":
    main()
