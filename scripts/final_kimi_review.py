"""Final review via Kimi k3: chunked text (v1.6) + figures, retry-safe."""
import base64, time
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI

kv = dict(l.strip().split("=", 1) for l in open(r"secrets.env", encoding="utf-8").read().splitlines() if "=" in l)
client = OpenAI(api_key=kv["KIMI_API_KEY"], base_url=kv["KIMI_BASE_URL"] + "/v1", timeout=300)
BASE = Path(r".")
TXT = (BASE / "review/materials_v16/paper_full_v16.txt").read_text(encoding="utf-8", errors="replace")

HEAD = """You are a senior reviewer and area editor for Advanced Engineering Informatics (Elsevier, Q1). Manuscript: "When Objects Overshadow Materials: Benchmarking and Adapting Multimodal Large Language Models for Construction and Demolition Waste Auditing" (v1.6, 32 pages) — CDW-VQA, the first VQA benchmark for construction and demolition waste, built from public segmentation datasets (CODD, CDW-Seg); zero-shot evaluation of five model families; corrected-gate interaction analysis; exploratory error typology (object-prior hijack secondary to wrong-material); QLoRA adaptation for two Qwen models plus a GLM-4.1V family-split result; L0-only ablation; oracle-box diagnosis; supervised segmentation baseline; human baseline v2 (balanced blind protocol).

The full text is split into 3 chunks (this is chunk {chunk} of 3). Review with strict, evidence-backed comments (cite section/table/figure evidence). Produce sections:
1. Summary & claimed contributions — delivered?
2. Scope fit for AEI
3. Data authenticity & internal number consistency (abstract vs text vs figures vs tables)
4. Technical soundness: four-arm synthesis, statistics (McNemar, clustered bootstrap, TOST, failed pre-registered gate + corrected post-hoc contrast), QLoRA recipe
5. Structure/logic/language/length vs published AEI benchmark papers (AECBench)
6. Novelty & significance (honest; compare AECBench, Waste-Bench, LAS&T)
7. Major comments (numbered), Minor comments (numbered) — ONLY for content in this chunk
8. Provisional recommendation so far (you will finalise after chunk 3)

CHUNK {chunk}/3:
"""

chunks = [TXT[i:i+20000] for i in range(0, len(TXT), 20000)]
print(f"chunks: {len(chunks)}, sizes: {[len(c) for c in chunks]}", flush=True)

def call(msgs, tag, tries=3, max_tokens=8000):
    for t in range(tries):
        try:
            r = client.chat.completions.create(model="k3", max_tokens=max_tokens, messages=msgs)
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
            print(f"[{tag}] empty text, retry {t+1}", flush=True)
        except Exception as e:
            print(f"[{tag}] attempt {t+1} error: {type(e).__name__}: {str(e)[:150]}", flush=True)
        time.sleep(6 * (t + 1))
    return f"__FAILED__ {tag}"

out = {}
for i, c in enumerate(chunks, 1):
    print(f"=== chunk {i} ===", flush=True)
    out[f"chunk{i}"] = call([{"role": "user", "content": HEAD.format(chunk=i) + c}], f"chunk{i}")
    print(out[f"chunk{i}"][:300], flush=True)

FIG_PROMPT = """You are reviewing the figures of the AEI manuscript "CDW-VQA" (final version). For EACH figure below, judge: information density, expressiveness, consistency with the paper's claims (numbers/labels), and any visual defects (cropping, overlap, misalignment, unreadable text). Then give an overall figure quality score /10 and list the top-3 figure-level fixes.

Figures (in order): fig1 framework pipeline; fig2 four-arm design; fig3 zero-shot capability bars; fig4 collapse curves; fig5 hijack case; fig6 adaptation anatomy; fig7 adaptation before/after bars."""

def b64img(path, max_side=800):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

print("=== figures review ===", flush=True)
figs = sorted((BASE / "review/materials_v16").glob("fig*.png"))
content = [{"type": "text", "text": FIG_PROMPT}]
for p in figs:
    content.append({"type": "image_url", "image_url": {"url": b64img(p)}})
out["figures"] = call([{"role": "user", "content": content}], "figures", max_tokens=4000)

report = "# Kimi k3 final review (v1.6 manuscript)\n\n"
for k in ["chunk1", "chunk2", "chunk3"]:
    report += f"\n\n## {k}\n\n" + out.get(k, "__MISSING__")
report += "\n\n## Figures\n\n" + out.get("figures", "__MISSING__")
(BASE / "review" / "review_kimi_final.md").write_text(report, encoding="utf-8")
print("saved review_kimi_final.md", flush=True)
