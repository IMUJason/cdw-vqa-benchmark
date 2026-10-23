"""Independent recomputation of all manuscript numbers from released data.

Does NOT import fine_metrics.py / api_policy.py: the scoring rules are
re-implemented from the manuscript prose, so a disagreement with the released
scripts shows up here as a disagreement.

Every number below appears in paper/main.tex; each is recomputed from the
released jsonl/csv and compared against the value hard-coded here
(transcribed from the manuscript). MISMATCH lines are the audit output.

Files are located by glob so that no model-name literal is needed.

Run:  python scripts/audit_v17_recompute.py
"""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
DATA = BASE / "data"

ARMS = ["L0", "L1", "L2", "L3"]
FINDINGS = []


def note(ok, label, got, want, tol=0.0006):
    if isinstance(got, float) and isinstance(want, float):
        ok = abs(got - want) <= tol
    if not ok:
        FINDINGS.append((label, got, want))
    print("  [%s] %s: recomputed=%s manuscript=%s"
          % ("OK   " if ok else "MISMATCH", label, got, want))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def one(dirpath, pattern):
    """Exactly one file matching pattern."""
    hits = sorted(Path(dirpath).glob(pattern))
    if len(hits) != 1:
        raise SystemExit("expected 1 hit for %s in %s, got %s"
                         % (pattern, dirpath, [h.name for h in hits]))
    return hits[0]


def strip_md(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</?answer>", "", s)
    s = re.sub(r"\*+", "", s)
    return s.strip().lower()


OBJ_PIPE = r"\b(pipe|pipes|piping|tube|tubes|hose|hoses|conduit|tubing)\b"
FINE_PVC = r"\b(pvc|upvc|cpvc|vinyl|polyvinyl\s+chloride)\b"
FINE_METAL = (r"\b(metal|metallic|steel|iron|copper|aluminum|aluminium|brass|"
              r"galvanis?zed)\b")
COARSE_PVC_EXTRA = r"\b(plastic|plastics|polymer|polypropylene|polyethylene)\b"

# Delivery policy for API models (mirrors scripts/api_policy.py): non-delivery
# iff raw is empty or an explicit cannot-see-the-image statement. NOTE:
# api_delivery_mask.json is keyed by uid and its two model loops overwrite
# each other, so the mask holds GLM-5.3's delivery bits only; delivery must
# be recomputed from raw with this regex instead.
UNABLE = re.compile(
    r"unable to (?:view|see|process|load|access|identify)|"
    r"(?:do(?:es)? not|don'?t|cannot|can'?t|am not able to|not able to) "
    r"(?:see|view|access|process|identify|determine) (?:any |an? |the )?image|"
    r"no image (?:is |was )?(?:attached|provided|included|visible|received)|"
    r"only (?:see|received) (?:a |an )?(?:link|text|placeholder)|"
    r"image (?:failed|failure) (?:to )?(?:upload|load)",
    re.I)


def delivered(raw):
    r = (raw or "").strip()
    return bool(r) and not UNABLE.search(r)

def obj_hit(raw):
    return bool(re.search(OBJ_PIPE, strip_md(raw)))


def mat_hit(raw, truth):
    a = strip_md(raw)
    if truth == "PVC":
        fine = bool(re.search(FINE_PVC, a))
        coarse = fine or bool(re.search(COARSE_PVC_EXTRA, a))
    else:
        fine = bool(re.search(FINE_METAL, a))
        coarse = fine
    return fine, coarse


def pipes(rows, split=None):
    """Scored pipes items: the 348 instances whose material truth is
    PVC or metal (the 2 'other' pipes are excluded from scoring, giving
    the 3,872 = 3,880 - 8 evaluated-item count in Section 3.5).

    Object rows are scored over all pipes object items (answer holds the
    class name, so an "other"-material pipe still has a valid object
    truth): n=350 all-splits / n=189 test. Material rows additionally
    require answer in {PVC, metal}: n=348 / n=188. This is the
    manuscript's denominator convention (3,880 - 8 = 3,872 scored)."""
    out = []
    for r in rows:
        if r.get("cls") != "pipes" or r.get("needs_manual_label"):
            continue
        if split and r.get("split") != split:
            continue
        fam = r.get("family")
        if fam == "instance_object":
            out.append(r)
        elif fam == "instance_material" and r.get("answer") in ("PVC", "metal"):
            out.append(r)
    return out


def acc(vals):
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------- 1. suites
print("=" * 78)
print("1. SUITE ARITHMETIC (Sections 3.3-3.5, Table 2, Fig. 1)")
print("=" * 78)

scene = load(DATA / "benchmark_v1_scene_qa.jsonl")
note(len(scene) == 4072, "scene suite size = 4072", len(scene), 4072)
imgs = {r["image_path"] for r in scene}
codd_imgs = {r["image_path"] for r in scene if r["dataset"] == "codd"}
seg_imgs = {r["image_path"] for r in scene if r["dataset"] == "cdw_seg"}
note(len(imgs) == 930, "scene images (unique paths) = 930", len(imgs), 930)
note(len(codd_imgs) == 500, "CODD images = 500", len(codd_imgs), 500)
note(len(seg_imgs) == 430, "CDW-Seg images = 430", len(seg_imgs), 430)

by_ds_fam = Counter((r["dataset"], r["family"]) for r in scene)
print("  scene per-cell n: %s" % dict(sorted(by_ds_fam.items())))
for key, v in [(("codd", "F1_presence"), 1000), (("codd", "F2_extent"), 500),
               (("codd", "F3_dominance"), 83), (("codd", "F4_contamination"), 500),
               (("cdw_seg", "F1_presence"), 846),
               (("cdw_seg", "F2_extent"), 425),
               (("cdw_seg", "F3_dominance"), 288),
               (("cdw_seg", "F4_contamination"), 430)]:
    note(by_ds_fam.get(key, 0) == v, "scene cell n %s/%s" % key,
         by_ds_fam.get(key, 0), v)

inst = load(DATA / "diagnostic_v2_qa.jsonl")
note(len(inst) == 3880, "instance suite nominal 485x4x2 = 3880", len(inst), 3880)
n_pipes_rows = sum(1 for r in inst if r.get("cls") == "pipes")
n_eval = sum(1 for r in inst
             if r.get("cls") == "pipes" and not r.get("needs_manual_label"))
note(n_pipes_rows - n_eval == 8, "pipes other exclusions = 8",
     n_pipes_rows - n_eval, 8)
n_diag = len({r["diag_id"] for r in inst})
note(n_diag == 485, "diagnostic instances = 485", n_diag, 485)
cls_counts = Counter(r.get("cls") for r in inst if r.get("arm") == "L0"
                      and r.get("family") == "instance_object")
print("  instance class mix (L0/object): %s" % dict(cls_counts))
split_counts = Counter()
seen = set()
for r in inst:
    if r["diag_id"] in seen:
        continue
    seen.add(r["diag_id"])
    split_counts[r["split"]] += 1
print("  instance split sizes: %s" % dict(split_counts))
note(split_counts["train"] == 194 and split_counts["val"] == 64
     and split_counts["test"] == 227, "instance split = 194/64/227",
     "%d/%d/%d" % (split_counts["train"], split_counts["val"],
                   split_counts["test"]), "194/64/227")

sft = load(DATA / "sft_train.jsonl")
note(len(sft) == 4277, "SFT instruction examples = 4277", len(sft), 4277)
sft_scene = Counter(r["family"] for r in sft if r.get("source") == "scene")
sft_inst = Counter(r["family"] for r in sft if r.get("source") == "instance")
print("  SFT scene families: %s" % dict(sft_scene))
print("  SFT instance families: %s" % dict(sft_inst))
for fam, v in [("F1_presence", 1230), ("F2_extent", 617),
               ("F3_dominance", 261), ("F4_contamination", 621)]:
    note(sft_scene.get(fam) == v, "SFT scene %s" % fam, sft_scene.get(fam), v)
note(sft_inst.get("instance_object") == 776, "SFT instance object",
     sft_inst.get("instance_object"), 776)
note(sft_inst.get("instance_material") == 772, "SFT instance material",
     sft_inst.get("instance_material"), 772)
arm_mix = Counter(r["image"].split("_")[-1].replace(".jpg", "") for r in sft
                  if r.get("source") == "instance")
print("  SFT instance per-arm: %s" % dict(sorted(arm_mix.items())))

# ---------------------------------------------------------------- 2. truth
print()
print("=" * 78)
print("2. PIPES GROUND TRUTH (Section 3.6)")
print("=" * 78)
final = list(csv.DictReader(
    open(DATA / "pipes_material_final.csv", encoding="utf-8-sig")))
mat_counts = Counter(r["material"] for r in final)
print("  final pipes truth: %s total=%d" % (dict(mat_counts), len(final)))
note(len(final) == 350, "pipes final labels = 350", len(final), 350)
note(mat_counts.get("PVC") == 308, "PVC = 308", mat_counts.get("PVC"), 308)
note(mat_counts.get("metal") == 40, "metal = 40", mat_counts.get("metal"), 40)
note(mat_counts.get("other", 0) == 2, "other = 2", mat_counts.get("other", 0), 2)
n_scored = mat_counts["PVC"] + mat_counts["metal"]
pp = mat_counts["PVC"] / n_scored
H = -(pp * math.log2(pp) + (1 - pp) * math.log2(1 - pp))
note(abs(H - 0.515) <= 0.001, "H(m|pipes) = 0.515 bits", round(H, 4), 0.515)
note(abs(pp - 0.885) <= 0.001, "PVC base rate = 88.5%", round(pp, 4), 0.885)

# ---------------------------------------------------------------- 3. zero-shot
print()
print("=" * 78)
print("3. ZERO-SHOT COLLAPSE (Section 4.3, Fig. 4)")
print("=" * 78)
ZS_FILES = [
    ("q3", one(RES / "w6", "responses_qwen3*.jsonl")),
    ("q25", one(RES / "w6", "responses_qwen2*.jsonl")),
    ("g", one(RES / "w6", "responses_g*.jsonl")),
]
ZS_OBJ = {  # 350-denominator values (all pipes object items incl. the 2
    # 'other' pipes), matching fig.4, Table 4 rows, and stored 'correct';
    # object-question denominators are n=350 (all pipes score on object questions)
    "q3": (0.343, 0.094, 0.114, 0.206),
    "g": (0.246, 0.054, 0.037, 0.094),
    "q25": (0.120, 0.043, 0.031, 0.057),
}
ZS_COARSE = {"q3": (0.500, 0.109), "g": (0.394, 0.112),
             "q25": (0.477, 0.141)}
for mk, path in ZS_FILES:
    rows = pipes(load(path))
    for i, arm in enumerate(ARMS):
        sel = [r for r in rows
               if r["family"] == "instance_object" and r["arm"] == arm]
        got = acc([obj_hit(r["raw"]) for r in sel])
        note(abs(got - ZS_OBJ[mk][i]) <= 0.001, "%s %s object" % (mk, arm),
             round(got, 4), ZS_OBJ[mk][i])
    for i, arm in enumerate(["L0", "L1"]):
        sel = [r for r in rows
               if r["family"] == "instance_material" and r["arm"] == arm]
        got = acc([mat_hit(r["raw"], r["answer"])[1] for r in sel])
        note(abs(got - ZS_COARSE[mk][i]) <= 0.001,
             "%s %s coarse material" % (mk, arm), round(got, 4),
             ZS_COARSE[mk][i])
    fins = []
    for arm in ARMS:
        sel = [r for r in rows
               if r["family"] == "instance_material" and r["arm"] == arm]
        fins.append(round(acc([mat_hit(r["raw"], r["answer"])[0] for r in sel]), 3))
    print("    %s fine material per arm: %s (manuscript floor 0.017-0.083)"
          % (mk, fins))

# ---------------------------------------------------------------- 4. gate
print()
print("=" * 78)
print("4. CORRECTED INTERACTION GATE (Section 4.6)")
print("=" * 78)
INT = {"q3": 0.313, "g": 0.244, "q25": 0.141}
per_model = {}
for mk, path in ZS_FILES:
    rows = pipes(load(path))
    d = {}
    for arm in ARMS:
        o = {r["diag_id"]: obj_hit(r["raw"]) for r in rows
             if r["family"] == "instance_object" and r["arm"] == arm}
        m = {r["diag_id"]: mat_hit(r["raw"], r["answer"])[0] for r in rows
             if r["family"] == "instance_material" and r["arm"] == arm}
        ids = sorted(set(o) & set(m))
        d[arm] = (acc([o[i] for i in ids]), acc([m[i] for i in ids]))
    inter = (d["L0"][0] - d["L0"][1]) - (d["L1"][0] - d["L1"][1])
    per_model[mk] = inter
    note(abs(inter - INT[mk]) <= 0.001, "%s corrected interaction" % mk,
         round(inter, 4), INT[mk])
pooled = sum(per_model.values()) / 3
note(abs(pooled - 0.233) <= 0.001, "pooled corrected interaction",
     round(pooled, 4), 0.233)

# ---------------------------------------------------------------- 5. adaptation
print()
print("=" * 78)
print("5. ADAPTATION (Sections 5.2-5.3, Fig. 7)")
print("=" * 78)
AD_FILES = [
    ("q3", one(RES / "adapted", "adapted_qwen3*.jsonl")),
    ("q25", one(RES / "adapted", "adapted_qwen25*.jsonl")),
]
AD_OBJ = {
    "q3": {"L0": 0.487, "L1": 0.862, "L2": 0.841, "L3": 0.926},
    "q25": {"L0": 0.899, "L1": 0.947, "L2": 0.894, "L3": 0.947},
}
AD_FINE = {"q3": {"L0": 0.378, "L1": 0.766},
           "q25": {"L0": 0.622, "L1": 0.840}}
for mk, path in AD_FILES:
    rows = pipes(load(path), split="test")
    for arm in ARMS:
        sel = [r for r in rows
               if r["family"] == "instance_object" and r["arm"] == arm]
        got = acc([obj_hit(r["raw"]) for r in sel])
        note(abs(got - AD_OBJ[mk][arm]) <= 0.002,
             "%s adapted %s object" % (mk, arm), round(got, 4),
             AD_OBJ[mk][arm])
    for arm in ["L0", "L1"]:
        sel = [r for r in rows
               if r["family"] == "instance_material" and r["arm"] == arm]
        got = acc([mat_hit(r["raw"], r["answer"])[0] for r in sel])
        note(abs(got - AD_FINE[mk][arm]) <= 0.002,
             "%s adapted %s fine material" % (mk, arm), round(got, 4),
             AD_FINE[mk][arm])

print("    (manuscript: object 0.06-0.12, fine 0.01-0.07)")
rows = pipes(load(one(RES / "adapted", "adapted_g*.jsonl")), split="test")
for arm in ARMS:
    so = [r for r in rows
          if r["family"] == "instance_object" and r["arm"] == arm]
    sm = [r for r in rows
          if r["family"] == "instance_material" and r["arm"] == arm]
    o = acc([obj_hit(r["raw"]) for r in so]) if so else float("nan")
    f = acc([mat_hit(r["raw"], r["answer"])[0] for r in sm]) if sm else float("nan")
    print("      %s: object=%.3f fine=%.3f" % (arm, o, f))

# ---------------------------------------------------------------- 6. oracle
print()
print("=" * 78)
print("6. ORACLE-BOX DIAGNOSTIC (Section 5.3)")
print("=" * 78)
ORC_FILES = [
    ("q3", one(RES / "oracle", "oracle_boxed_qwen3*.jsonl")),
    ("q25", one(RES / "oracle", "oracle_boxed_qwen25*.jsonl")),
]
ORC = {"q3": {"obj": 0.862, "fine": 0.777},
       "q25": {"obj": 0.952, "fine": 0.750}}
for mk, path in ORC_FILES:
    rows = pipes(load(path), split="test")
    sel = [r for r in rows
           if r["family"] == "instance_object" and r["arm"] == "L0"]
    got = acc([obj_hit(r["raw"]) for r in sel])
    note(abs(got - ORC[mk]["obj"]) <= 0.002, "%s oracle L0 object" % mk,
         round(got, 4), ORC[mk]["obj"])
    sel = [r for r in rows
           if r["family"] == "instance_material" and r["arm"] == "L0"]
    got = acc([mat_hit(r["raw"], r["answer"])[0] for r in sel])
    note(abs(got - ORC[mk]["fine"]) <= 0.002, "%s oracle L0 fine material" % mk,
         round(got, 4), ORC[mk]["fine"])

# ---------------------------------------------------------------- 7. ablation
print()
print("=" * 78)
print("7. L0-ONLY ABLATION (Section 5.3)")
print("=" * 78)
L0AB = {"q3": (0.815, 0.612), "q25": (0.704, 0.782)}
L0_FILES = [
    ("q3", one(RES / "adapted_l0", "l0ab_qwen3*.jsonl")),
    ("q25", one(RES / "adapted_l0", "l0ab_qwen25*.jsonl")),
]
for mk, path in L0_FILES:
    rows = pipes(load(path), split="test")
    sel = [r for r in rows
           if r["family"] == "instance_object" and r["arm"] == "L1"]
    got = acc([obj_hit(r["raw"]) for r in sel])
    note(abs(got - L0AB[mk][0]) <= 0.002, "%s L0-only object L1" % mk,
         round(got, 4), L0AB[mk][0])
    sel = [r for r in rows
           if r["family"] == "instance_material" and r["arm"] == "L1"]
    got = acc([mat_hit(r["raw"], r["answer"])[0] for r in sel])
    note(abs(got - L0AB[mk][1]) <= 0.002, "%s L0-only fine material L1" % mk,
         round(got, 4), L0AB[mk][1])

# ---------------------------------------------------------------- 8. scene
print()
print("=" * 78)
print("8. SCENE CAPABILITY MAP (Section 4.2, Fig. 3)")
print("=" * 78)
SC_FILES = [
    ("q3", one(RES / "scene_eval", "scene_qwen3*.jsonl")),
    ("g", one(RES / "scene_eval", "scene_g*.jsonl")),
    ("q25", one(RES / "scene_eval", "scene_qwen2*.jsonl")),
]


def yesno_correct(raw, truth):
    a = strip_md(raw)
    m = re.match(r"^\W*(yes|no)\b", a)
    if not m:
        return None
    return m.group(1) == truth.strip().lower()


def mcq_correct(raw, truth):
    a = strip_md(raw)
    m = re.match(r"^\W*([a-e])\b", a)
    if not m:
        return None
    return m.group(1) == truth.strip().lower()


for mk, path in SC_FILES:
    rows = load(path)
    for ds in ["codd", "cdw_seg"]:
        for fam in ["F1_presence", "F2_counting", "F3_dominance",
                    "F4_contamination"]:
            sel = [r for r in rows
                   if r["dataset"] == ds and r["family"] == fam]
            if not sel:
                continue
            if fam in ("F1_presence", "F4_contamination"):
                vals = [yesno_correct(r["raw"], r["answer"]) for r in sel]
            else:
                vals = [mcq_correct(r["raw"], r["answer"]) for r in sel]
            got = acc([v for v in vals if v is not None])
            print("    %-5s %-8s %-18s n=%4d acc=%.3f (parsed %d)"
                  % (mk, ds, fam, len(sel), got,
                     sum(v is not None for v in vals)))
    sel = [r for r in rows if r["family"] == "F4_contamination"]
    yes_rate = acc([r["answer"].strip().lower() == "yes" for r in sel])
    print("    %-5s F4 Yes base rate = %.3f (manuscript 0.77)" % (mk, yes_rate))
    for ds in ["codd", "cdw_seg"]:
        sel = [r for r in rows
               if r["dataset"] == ds and r["family"] == "F4_contamination"]
        rec = {"yes": [], "no": []}
        for r in sel:
            v = yesno_correct(r["raw"], r["answer"])
            if v is None:
                continue
            rec[r["answer"].strip().lower()].append(v)
        yr = acc(rec["yes"]) if rec["yes"] else float("nan")
        nr = acc(rec["no"]) if rec["no"] else float("nan")
        print("    %-5s F4 %-8s balanced = %.3f (yes %.3f / no %.3f)"
              % (mk, ds, (yr + nr) / 2, yr, nr))

# ------------------------------------------------------------ 9. API table
print()
print("=" * 78)
print("9. FRONTIER API TABLE (Table 4, delivery policy Section 3.7)")
print("=" * 78)
API = {"kimi": "api_kimi.jsonl", "": "api_glm.jsonl"}
TAB = {  # manuscript Table 4: success, F3 codd e2e/cond, F3 cdw e2e/cond,
         # F2 cdw e2e/cond, pipes obj/cond and coarse/cond per arm, fine L0
    "kimi": dict(success=.838, f3_codd=(.313, .667), f3_seg=(.521, .564),
                 f2_seg=(.106, .479),
                 pipes={a: None for a in ARMS},
                 pipes_row={("L0"): (.319, .348), ("L1"): (.084, .124),
                            ("L2"): (.081, .152), ("L3"): (.101, .114)},
                 fine_L0=.014),
    "": dict(success=.881, f3_codd=(.530, .543), f3_seg=(.299, .323),
                f2_seg=(.365, .388),
                pipes_row={("L0"): (.000, .013), ("L1"): (.000, .033),
                           ("L2"): (.000, .039), ("L3"): (.000, .020)},
                fine_L0=.003),
}
for mk, f in API.items():
    recs = load(RES / "api_eval" / f)
    inst = [r for r in recs if r.get("level") != "scene"
            and r.get("cls") == "pipes"]
    scene = [r for r in recs if r.get("level") == "scene"]
    succ = acc([delivered(r["raw"]) for r in recs])
    note(abs(succ - TAB[mk]["success"]) <= 0.002, "%s delivery success" % mk,
         round(succ, 4), TAB[mk]["success"])
    for fam, ds, key in [("F3_dominance", "codd", "f3_codd"),
                         ("F3_dominance", "cdw_seg", "f3_seg"),
                         ("F2_extent", "cdw_seg", "f2_seg")]:
        sel = [r for r in scene if r["family"] == fam and r["dataset"] == ds]
        e2e = acc([mcq_correct(r["raw"], r["answer"]) or False for r in sel])
        dv = [r for r in sel if delivered(r["raw"])]
        cn = acc([mcq_correct(r["raw"], r["answer"]) or False for r in dv])
        want = TAB[mk][key]
        note(abs(e2e - want[0]) <= 0.002 and abs(cn - want[1]) <= 0.002,
             "%s %s %s e2e/cond" % (mk, fam, ds), "%.3f/%.3f" % (e2e, cn),
             "%.3f/%.3f" % want)
    for arm in ARMS:
        so = [r for r in inst if r["family"] == "instance_object"
              and r["arm"] == arm and delivered(r["raw"])]
        sm = [r for r in inst if r["family"] == "instance_material"
              and r["arm"] == arm and r.get("answer") in ("PVC", "metal")
              and delivered(r["raw"])]
        o = acc([obj_hit(r["raw"]) for r in so])
        m = acc([mat_hit(r["raw"], r["answer"])[1] for r in sm])
        want = TAB[mk]["pipes_row"][arm]
        note(abs(o - want[0]) <= 0.002 and abs(m - want[1]) <= 0.002,
             "%s pipes %s obj/cond coarse/cond" % (mk, arm),
             "%.3f/%.3f" % (o, m), "%.3f/%.3f" % want)
    smf = [r for r in inst if r["family"] == "instance_material"
           and r["arm"] == "L0" and r.get("answer") in ("PVC", "metal")
           and delivered(r["raw"])]
    fin = acc([mat_hit(r["raw"], r["answer"])[0] for r in smf])
    note(abs(fin - TAB[mk]["fine_L0"]) <= 0.002, "%s fine L0" % mk,
         round(fin, 4), TAB[mk]["fine_L0"])
print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
if not FINDINGS:
    print("  NO MISMATCHES: every manuscript number recomputed from released "
          "data matches.")
else:
    print("  %d MISMATCH(ES):" % len(FINDINGS))
    for label, got, want in FINDINGS:
        print("    - %s: recomputed=%s manuscript=%s" % (label, got, want))
