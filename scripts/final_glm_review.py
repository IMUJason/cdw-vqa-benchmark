"""Final review via GLM-5.3 (stable endpoint): text + figures, chunked, retry-safe."""
import json, base64, time
from pathlib import Path
import cv2
import numpy as np
import requests

kv = dict(l.strip().split("=", 1) for l in open(r"secrets.env", encoding="utf-8").read().splitlines() if "=" in l)
URL = kv["GLM_BASE_URL"] + "/v1/messages"
HDR = {"x-api-key": kv["GLM_API_KEY"], "anthropic-version": "2023-06-01"}
BASE = Path(r".")
TXT = (BASE / "review/materials/paper_full.txt").read_text(encoding="utf-8", errors="replace")

PROMPT_MAIN = """You are a senior reviewer and area editor for Advanced Engineering Informatics (Elsevier, Q1). Manuscript: CDW-VQA, the first VQA benchmark for construction and demolition waste, zero-shot evaluation of five model families, error-typology (object-prior hijack), QLoRA adaptation study.

Review the FULL text (pdftotext, 24 pages) and produce a strict, evidence-backed structured review (cite section/line evidence):
1. Summary & claimed contributions delivered?
2. Scope fit for AEI
3. Data authenticity & internal number consistency (abstract vs text vs figures vs tables)
4. Technical soundness: algorithms (four-arm), statistics (McNemar/bootstrap/TOST, failed pre-registered gap), QLoRA recipe, param arithmetic (43.6M/0.50% vs 47.6M/0.57%)
5. Structure/logic/language/length vs published AEI benchmark papers (AECBench)
6. Novelty & significance (honest; compare AECBench, Waste-Bench, LAS&T)
7. Major comments (numbered), Minor comments (numbered)
8. Recommendation (accept/minor/major/reject) with 3-sentence justification.

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
    return base64.b64encode(buf.tobytes()).decode()


def call(blocks, tag, tries=3):
    for t in range(tries):
        try:
            r = requests.post(URL, headers=HDR, json={
                "model": "glm-5.3", "max_tokens": 8000,
                "messages": [{"role": "user", "content": blocks}]}, timeout=300)
            r.raise_for_status()
            d = r.json()
            txt = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            if txt.strip():
                return txt.strip()
            print(f"[{tag}] empty text, retry {t+1}", flush=True)
        except Exception as e:
            print(f"[{tag}] attempt {t+1} error: {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(4 * (t + 1))
    return f"__FAILED__ {tag}"


out = {}
out["main"] = call([{"type": "text", "text": PROMPT_MAIN}], "main")

FIG_PROMPT = """Review these 7 figures from the CDW-VQA manuscript (order: fig1 framework pipeline; fig2 four-arm design; fig3 zero-shot capability bars; fig4 collapse curves; fig5 hijack case; fig6 adaptation anatomy; fig7 adaptation before/after bars). For each: information density, expressiveness, visual defects (cropping/overlap/misalignment/unreadable). Give an overall figure score /10 and top-3 fixes."""
content = [{"type": "text", "text": FIG_PROMPT}]
for p in sorted((BASE / "review/materials").glob("fig*.png")):
    content.append({"type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64img(p)}})
out["figures"] = call(content, "figures")

(BASE / "review" / "review_glm53_final.md").write_text(
    "# GLM-5.3 final review\n\n## Main text\n\n" + out["main"] +
    "\n\n## Figures\n\n" + out["figures"], encoding="utf-8")
print("saved review_glm53_final.md")
print(out["main"][:600])
