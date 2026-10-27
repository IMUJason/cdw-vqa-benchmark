"""Unified scene-answer extraction and scoring (released scorer).

Rule
----
A model's verdict is the content of its *boxed answer channel* when the model
emits one (`<|begin_of_box|> X <|end_of_box|>`); otherwise it is the first answer
token after stripping markdown decoration and answer tags. A response with
neither a boxed verdict nor a parseable first token is counted as an error
(non-termination), and its rate is reported separately.

Why: three of the five evaluated systems answer in the first token, so the two
rules coincide for them (box share 0.00). GLM-4.1V-9B-Thinking writes a reasoning
preamble and then a boxed verdict (box share 0.91-0.99); a first-token-only rule
would score its format, not its answers. This script applies one rule to all
systems and reports the strict-first-token score alongside, so the difference is
visible rather than buried in a parser.

Usage:
    python scripts/rescore_scene.py            # table to stdout + JSON
Writes results/scene_eval/scene_scores_unified.json
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SC = BASE / "results" / "scene_eval"
AE = BASE / "results" / "api_eval"

ALIAS = {"timber": "wood", "plasterboard": "gypsum board", "gypsum_board": "gypsum board",
         "hard_plastic": "plastics", "soft_plastic": "plastics", "plastic": "plastics",
         "general_w": "general mixed", "fill_dirt": "fill dirt"}

BOX = re.compile(r"<\|begin_of_box\|>\s*\(?\s*([A-Da-d]|yes|no)\b", re.I)


def norm(s):
    s = str(s).strip().lower().replace("_", " ")
    return ALIAS.get(s, s)


def strip_decoration(a):
    a = re.sub(r"<think>.*?</think>", "", a or "", flags=re.S)
    a = re.sub(r"<answer>|</answer>|<\|begin_of_box\|>|<\|end_of_box\|>", "", a)
    return re.sub(r"\*+", "", a).strip()


def box_token(r):
    ms = list(BOX.finditer(re.sub(r"<think>.*?</think>", "", r.get("raw") or "", flags=re.S)))
    if not ms:
        return None
    v = ms[-1].group(1)
    return v.upper() if len(v) == 1 else v.lower()


def first_token(r):
    a = strip_decoration(r.get("raw"))
    if r["qtype"] == "yesno":
        letters = re.sub(r"[^a-z]", "", a.lower())
        if letters.startswith("yes"):
            return "yes"
        if letters.startswith("no"):
            return "no"
        return None
    m = re.match(r"\(?([a-dA-D])\b", a)
    return m.group(1).upper() if m else None


def unified(r):
    return box_token(r) or first_token(r)


def hit(r, v):
    if v is None:
        return False
    if r["qtype"] == "yesno":
        return (v == "yes") == str(r["answer"]).strip().lower().startswith("y")
    ch = (r.get("choices") or {}).get(v.upper())
    return ch is not None and norm(ch) == norm(r.get("answer_text"))


def evaluate(rows, split=None):
    out = {}
    for fam in ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]:
        for ds in ["codd", "cdw_seg"]:
            sel = [r for r in rows if r["family"] == fam and r.get("dataset") == ds
                   and (split is None or r.get("split") in split)]
            if not sel:
                continue
            out[f"{fam}|{ds}"] = dict(
                n=len(sel),
                unified=sum(hit(r, unified(r)) for r in sel) / len(sel),
                first=sum(hit(r, first_token(r)) for r in sel) / len(sel),
                no_verdict=sum(1 for r in sel if unified(r) is None) / len(sel),
                box_share=sum(1 for r in sel if box_token(r) is not None) / len(sel))
    return out


def main():
    files = {
        "qwen3-vl": SC / "scene_qwen3-vl-8b.jsonl",
        "qwen2.5-vl": SC / "scene_qwen2.5-vl-7b.jsonl",
        "glm-4.1v": SC / "scene_glm-4.1v-9b.jsonl",
        "kimi-k3": AE / "api_kimi_v2.jsonl",
        "glm-5.3": AE / "api_glm_v2.jsonl",
    }
    report = {}
    for mk, p in files.items():
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if mk in ("kimi-k3", "glm-5.3"):
            rows = [r for r in rows if r.get("level") == "scene"]
        report[mk] = {"all": evaluate(rows), "test": evaluate(rows, split={"test", "testing"})}
        rep = report[mk]["all"]
        print(f"\n== {mk} (all splits)")
        print(f"   {'family':18s} {'CODD uni':>9s} {'CODD first':>10s} "
              f"{'CDW uni':>8s} {'CDW first':>9s} {'no-verdict':>10s} {'box':>5s}")
        for fam in ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]:
            a = rep.get(f"{fam}|codd")
            b = rep.get(f"{fam}|cdw_seg")
            if not a or not b:
                continue
            print(f"   {fam:18s} {a['unified']:9.3f} {a['first']:10.3f} "
                  f"{b['unified']:8.3f} {b['first']:9.3f} "
                  f"{(a['no_verdict']+b['no_verdict'])/2:10.1%} {a['box_share']:5.2f}")
    (SC / "scene_scores_unified.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE results/scene_eval/scene_scores_unified.json")


if __name__ == "__main__":
    main()
