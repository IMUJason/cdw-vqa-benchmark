"""Unified delivery policy AND re-scoring for API models (Kimi-k3, GLM-5.3).

The stored `correct` field was produced by a first-cut parser that does not
strip markdown decoration, so it systematically mis-scores GLM-5.3
("**Yes.** The ...") on yes/no items. This script re-scores BOTH models under
one identical policy so Table numbers are reproducible from the raw files:

Delivery: non-delivery iff raw empty, or an explicit cannot-see-the-image
statement (wide regex, incl. "don't see an image", "I'm not able to view").

Scoring:
  scene yesno : first yes/no word after stripping non-letters
  scene mcq   : first a-d letter after stripping markdown
  instance object  : symmetric pipe-noun alias set (as fine_metrics)
  instance material: two levels -- fine (PVC/vinyl vs metal nouns) and
                     coarse (plastic-family counts as PVC)

For every cell: success rate, end-to-end accuracy, conditional accuracy.
Writes results/api_eval/api_policy.txt and api_delivery_mask.json.
"""
import json, re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "api_eval"

UNABLE = re.compile(
    r"unable to (?:view|see|process|load|access|identify)|"
    r"(?:do(?:es)? not|don'?t|cannot|can'?t|am not able to|not able to) "
    r"(?:see|view|access|process|identify|determine) (?:any |an? |the )?image|"
    r"no image (?:is |was )?(?:attached|provided|included|visible|received)|"
    r"only (?:see|received) (?:a |an )?(?:link|text|placeholder)|"
    r"image (?:failed|failure) (?:to )?(?:upload|load)",
    re.I)

OBJ = r"\b(pipe|pipes|piping|tube|tubes|hose|hoses|conduit|tubing)\b"
FINE = {
    "PVC": r"\b(pvc|upvc|cpvc|vinyl|polyvinyl\s+chloride)\b",
    "metal": r"\b(metal|metallic|steel|iron|copper|aluminum|aluminium|brass|galvanis?zed)\b",
}
COARSE_EXTRA_PVC = r"\b(plastic|plastics|polymer|polypropylene|polyethylene)\b"


def clean(a: str) -> str:
    a = re.sub(r"<think>.*?</think>", "", a or "", flags=re.S)
    a = re.sub(r"\*+", "", a)
    return a.strip().lower()


def delivered(raw: str) -> bool:
    r = (raw or "").strip()
    if not r:
        return False
    return not UNABLE.search(r)


def score_scene(r):
    """True/False, or None if no parseable answer (counts as error e2e)."""
    a = clean(r["raw"])
    if not a:
        return False
    if r.get("qtype") == "yesno":
        w = re.sub(r"[^a-z]", "", a)
        if w.startswith("yes"):
            return r["answer"].strip().lower().startswith("y")
        if w.startswith("no"):
            return not r["answer"].strip().lower().startswith("y")
        return False
    if r.get("qtype") == "mcq":
        a2 = a.replace("`", "")
        m = re.match(r"\(?([a-d])\b", a2)
        return bool(m and m.group(1) == r["answer"].strip().lower()[:1])
    return None


def score_instance(r, level):
    a = clean(r["raw"])
    if r["family"] == "instance_object":
        return bool(re.search(OBJ, a))
    truth = r["answer"]
    fine = bool(re.search(FINE[truth], a))
    coarse = fine or (truth == "PVC" and bool(re.search(COARSE_EXTRA_PVC, a)))
    return fine if level == "fine" else coarse


def main():
    lines = []
    P = lines.append
    mask = {}
    for name in ["kimi", "glm"]:
        rows = [json.loads(l) for l in open(RES / f"api_{name}.jsonl", encoding="utf-8")]
        n = len(rows)
        deliv = [r for r in rows if delivered(r["raw"])]
        for r in rows:
            mask[r["uid"]] = bool(delivered(r["raw"]))
        P("=" * 78)
        P(f"{name}: n={n} delivered={len(deliv)} success={len(deliv)/n:.3f} "
          f"non-delivery={n - len(deliv)} ({(n - len(deliv))/n:.1%})")
        P("=" * 78)

        def cell(sel, label, scorer):
            if not sel:
                return
            tot = len(sel)
            d = [r for r in sel if delivered(r["raw"])]
            c_all = sum(bool(scorer(r)) for r in sel)
            c_d = sum(bool(scorer(r)) for r in d)
            P(f"  {label:36s} n={tot:4d} e2e={c_all/tot:.3f} "
              f"cond={c_d/len(d):.3f} (n_del={len(d)})")

        for ds in ["codd", "cdw_seg"]:
            for fam in ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]:
                cell([r for r in rows if r.get("level") == "scene"
                      and r.get("dataset") == ds and r["family"] == fam],
                     f"scene {ds:8s} {fam[:2]}", score_scene)
        for lvl in ["coarse", "fine"]:
            for fam in ["instance_object", "instance_material"]:
                for arm in ["L0", "L1", "L2", "L3"]:
                    cell([r for r in rows if r.get("level") != "scene"
                          and r["family"] == fam and r.get("arm") == arm
                          and r.get("cls") == "pipes"],
                         f"pipes {fam[9:]:8s} {arm} [{lvl}]",
                         lambda r, lvl=lvl: score_instance(r, lvl) if fam == "instance_material"
                         else score_instance(r, "fine"))
        P("")
    txt = "\n".join(lines)
    (RES / "api_policy.txt").write_text(txt, encoding="utf-8")
    (RES / "api_delivery_mask.json").write_text(
        json.dumps(mask, indent=0), encoding="utf-8")
    print(txt)
    print("WROTE api_policy.txt / api_delivery_mask.json")


if __name__ == "__main__":
    main()
