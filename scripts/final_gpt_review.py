"""Final review via gmn relay GPT-6-astra: chunked text+figure review with retries."""
import json, base64, time, sys
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI

kv = dict(l.strip().split("=", 1) for l in open(r"secrets.env", encoding="utf-8").read().splitlines() if "=" in l)
client = OpenAI(api_key=kv["GPT_API_KEY"], base_url=kv["GPT_BASE_URL"] + "/v1", timeout=180)

BASE = Path(r".")
TXT = (BASE / "review/materials/paper_full.txt").read_text(encoding="utf-8", errors="replace")

PROMPT_MAIN = """You are a senior reviewer and area editor for Advanced Engineering Informatics (Elsevier, Q1). The manuscript is a benchmark paper: CDW-VQA, the first VQA benchmark for construction and demolition waste, with zero-shot evaluation of five model families, an error-typology analysis (object-prior hijack), and a QLoRA domain-adaptation study.

Review the FULL manuscript text below (pdftotext extraction, 24 pages) and produce a structured, evidence-backed review. Be strict and specific; cite page/section/line evidence for every criticism. Sections:
1. Summary and claimed contributions — does the paper deliver them?
2. Scope fit for Advanced Engineering Informatics
3. Data authenticity & internal number consistency (abstract vs text vs figures vs tables — check every cross-reference you can)
4. Technical soundness: algorithms (four-arm synthesis), statistics (McNemar, clustered bootstrap, TOST, the failed pre-registered gap analysis), QLoRA recipe, parameter arithmetic (43.6M/0.50% and 47.6M/0.57%)
5. Structure / logic / language / length vs published AEI benchmark papers (e.g., AECBench)
6. Novelty and significance — be honest, compare to AECBench, Waste-Bench, LAS&T
7. Major comments (numbered), Minor comments (numbered)
8. Recommendation: accept / minor / major / reject, with 3-sentence justification

MANUSCRIPT:
""" + TXT


def b64img(path, max_side=800):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def call(msgs, tag, tries=3):
    for t in range(tries):
        try:
            r = client.chat.completions.create(model=kv["GPT_MODEL"], max_tokens=6000, messages=msgs)
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception as e:
            print(f"[{tag}] attempt {t+1} error: {type(e).__name__}", flush=True)
            time.sleep(5 * (t + 1))
    return f"__FAILED__ {tag}"


out = {}
print("=== main text review ===")
out["main"] = call([{"role": "user", "content": PROMPT_MAIN}], "main")

FIG_PROMPT = """You are reviewing the figures of the AEI manuscript described in the previous review prompt (CDW-VQA benchmark). For EACH figure below, judge: information density, expressiveness, consistency with typical paper text, and any visual defects (cropping, overlap, misalignment, unreadable text). Then give an overall figure quality score /10 and list the top-3 figure-level fixes.

Figures (in order): fig1 framework pipeline; fig2 four-arm design; fig3 zero-shot capability bars; fig4 collapse curves; fig5 hijack case; fig6 adaptation anatomy; fig7 adaptation before/after bars."""

print("=== figures review ===")
figs = sorted((BASE / "review/materials").glob("fig*.png"))
content = [{"type": "text", "text": FIG_PROMPT}]
for p in figs:
    content.append({"type": "image_url", "image_url": {"url": b64img(p)}})
out["figures"] = call([{"role": "user", "content": content}], "figures")

(BASE / "review" / "review_gpt_final.md").write_text(
    "# GPT-6-astra final review\n\n## Main text\n\n" + out["main"] +
    "\n\n## Figures\n\n" + out["figures"], encoding="utf-8")
print("saved review_gpt_final.md")
print("--- MAIN (first 800 chars) ---")
print(out["main"][:800])
