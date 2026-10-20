"""Final review via Aimoniker Codex Auto gpt-6-astra (browser UA to pass Cloudflare)."""
import json, base64, time
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI
import httpx

kv = dict(l.strip().split("=", 1) for l in open(r"secrets.env", encoding="utf-8").read().splitlines() if "=" in l)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
client = OpenAI(api_key=kv["AIMONIKER_API_KEY"],
                base_url=kv["AIMONIKER_BASE_URL"],
                default_headers={"User-Agent": UA},
                http_client=httpx.Client(timeout=300, headers={"User-Agent": UA}))

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
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def call(msgs, tag, tries=4):
    for t in range(tries):
        try:
            r = client.chat.completions.create(model=kv["AIMONIKER_MODEL"],
                                               max_tokens=6000, messages=msgs)
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
            print(f"[{tag}] empty, retry {t+1}", flush=True)
        except Exception as e:
            print(f"[{tag}] attempt {t+1}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        time.sleep(6 * (t + 1))
    return f"__FAILED__ {tag}"


out = {}
print("=== main text review (chunked) ===", flush=True)
# 524 = CF 网关超时：单请求太大/太久。分三段送审，最后一段带汇总指令。
CACHE = BASE / "review" / "materials" / "aimoniker_parts.json"
CHUNKS = [TXT[i:i+20000] for i in range(0, len(TXT), 20000)]
part_reviews = None
if CACHE.exists():
    part_reviews = json.loads(CACHE.read_text(encoding="utf-8"))
    print("loaded cached part reviews", flush=True)
else:
    part_reviews = []
    for i, ch in enumerate(CHUNKS):
        p = (f"You are a strict AEI senior reviewer. This is PART {i+1}/{len(CHUNKS)} of the manuscript "
             "CDW-VQA (VQA benchmark for construction and demolition waste; five model families; "
             "object-prior hijack; QLoRA adaptation). Review THIS part: data/internal-number consistency, "
             "technical soundness, logic, language. List numbered issues with evidence; say n/a if none.\n\n" + ch)
        r = call([{"role": "user", "content": p}], f"part{i+1}")
        part_reviews.append(f"### Part {i+1}/{len(CHUNKS)}\n\n{r}")
        print(f"part {i+1} done ({len(r)} chars)", flush=True)
    CACHE.write_text(json.dumps(part_reviews, ensure_ascii=False, indent=1), encoding="utf-8")
out["main"] = ("## Chunked part reviews\n\n" + "\n\n".join(part_reviews) +
               "\n\n## Note: parts reviewed independently; overall verdict deferred to the figures+verdict call.")

VERDICT = """Based on the part reviews summarized below, plus these 7 figures (fig1 framework; fig2 four-arm; fig3 capability bars; fig4 collapse curves; fig5 hijack case; fig6 adaptation anatomy; fig7 adaptation before/after), give: (a) per-figure quality judgement and overall figure score /10 with top-3 fixes; (b) overall Major/Minor comments for the whole manuscript; (c) final recommendation (accept/minor/major/reject) with 3-sentence justification.

PART-REVIEW SUMMARIES:
""" + "\n".join(r[:1200] for r in part_reviews)
content = [{"type": "text", "text": VERDICT}]
for p in sorted((BASE / "review/materials").glob("fig*.png")):
    content.append({"type": "image_url", "image_url": {"url": b64img(p)}})
print("=== figures + verdict review ===", flush=True)
VERDICT = """Based on the part reviews summarized below, plus these 7 figures (fig1 framework; fig2 four-arm; fig3 capability bars; fig4 collapse curves; fig5 hijack case; fig6 adaptation anatomy; fig7 adaptation before/after), give: (a) per-figure quality judgement and overall figure score /10 with top-3 fixes; (b) overall Major/Minor comments for the whole manuscript; (c) final recommendation (accept/minor/major/reject) with 3-sentence justification.

PART-REVIEW SUMMARIES:
""" + "\n".join(r[:1200] for r in part_reviews)
out["figures"] = call(content, "figures+verdict")

(BASE / "review" / "review_gpt_aimoniker.md").write_text(
    "# GPT-6-astra (Aimoniker Codex Auto) final review\n\n## Main text\n\n" + out["main"] +
    "\n\n## Figures\n\n" + out["figures"], encoding="utf-8")
print("saved review_gpt_aimoniker.md")
print("--- MAIN head ---")
print(out["main"][:600])
