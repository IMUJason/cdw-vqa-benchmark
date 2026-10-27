"""Authoritative v2 number sheet (fine/coarse material scoring made explicit).

Writes results/v2_number_sheet.txt so manuscript edits read from one source.
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from fine_metrics import obj_score, mat_score

R = BASE / "results"
TEST = {"test", "testing"}
FAMS = ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]
LINES = []


def out(s=""):
    LINES.append(s)
    print(s)


def load(p):
    return [json.loads(l) for l in open(R / p, encoding="utf-8")]


def scene(rows, ds, fam, split="all"):
    sel = [r for r in rows if r.get("level") == "scene" and r.get("dataset") == ds
           and r["family"] == fam]
    if split == "test":
        sel = [r for r in sel if str(r.get("split", "")).lower() in TEST]
    return (sum(r["correct"] for r in sel) / len(sel), len(sel)) if sel else (None, 0)


def inst(rows, fam, arm, level="fine"):
    sel = [r for r in rows if r.get("family") == fam and r.get("arm") == arm
           and r.get("cls") == "pipes" and not r.get("needs_manual_label")]
    if not sel:
        return None, 0
    if fam == "instance_object":
        return sum(obj_score(r["raw"]) for r in sel) / len(sel), len(sel)
    return sum(mat_score(r["raw"], r["answer"])[0 if level == "fine" else 1]
               for r in sel) / len(sel), len(sel)


out("=== ZERO-SHOT (rebuilt keys) ===")
for name, p in [("Qwen3-VL", "scene_eval/scene_qwen3-vl-8b.jsonl"),
                ("Qwen2.5-VL", "scene_eval/scene_qwen2.5-vl-7b.jsonl"),
                ("GLM-4.1V", "scene_eval/scene_glm-4.1v-9b.jsonl")]:
    rows = load(p)
    for split in ("all", "test"):
        cells = []
        for ds in ("codd", "cdw_seg"):
            for f in FAMS:
                a, n = scene(rows, ds, f, split)
                cells.append(f"{ds[:4]}-{f[:2]}={a:.3f}(n{n})")
        out(f"{name:11s}[{split}] " + " ".join(cells))


out("\n=== ADAPTED v2 (test scene; instance all arms) ===")
for name, p in [("Qwen3-VL", "adapted/adapted_qwen3vl_v2.jsonl"),
                ("Qwen2.5-VL", "adapted/adapted_qwen25vl_v2.jsonl")]:
    rows = load(p)
    cells = []
    for ds in ("codd", "cdw_seg"):
        for f in FAMS:
            a, n = scene(rows, ds, f, "test")
            cells.append(f"{ds[:4]}-{f[:2]}={a:.3f}(n{n})")
    out(f"{name}: " + " ".join(cells))
    for fam in ("instance_object", "instance_material"):
        lvl = "fine"
        vals = []
        for arm in ("L0", "L1", "L2", "L3"):
            a, n = inst(rows, fam, arm, lvl)
            vals.append(f"{arm}={a:.3f}")
        out(f"   {fam} ({lvl}): " + " ".join(vals))

out("\n=== GLM-4.1V adapted (v1 adapter, unaffected families) ===")
g = load("adapted/adapted_glm41v.jsonl")
cells = []
for ds in ("codd", "cdw_seg"):
    for f in ("F1_presence", "F4_contamination"):
        a, n = scene(g, ds, f, "test")
        cells.append(f"{ds[:4]}-{f[:2]}={a:.3f}(n{n})")
out(" ".join(cells))
for fam in ("instance_object", "instance_material"):
    vals = [f"{a}={inst(g, fam, a)[0]:.3f}" for a in ("L0", "L1")]
    out(f"   {fam}: " + " ".join(vals))

out("\n=== ORACLE (v2, boxed L0) ===")
for name, p in [("Qwen3", "oracle/oracle_boxed_qwen3vl_v2.jsonl"),
                ("Qwen2.5", "oracle/oracle_boxed_qwen25vl_v2.jsonl")]:
    rows = load(p)
    o = [r for r in rows if r["family"] == "instance_object"]
    m = [r for r in rows if r["family"] == "instance_material"]
    out(f"{name}: object {sum(obj_score(r['raw']) for r in o)/len(o):.3f} (n={len(o)}) | "
        f"material fine {sum(mat_score(r['raw'],r['answer'])[0] for r in m)/len(m):.3f} "
        f"coarse {sum(mat_score(r['raw'],r['answer'])[1] for r in m)/len(m):.3f} (n={len(m)})")

out("\n=== L0-ONLY ABLATION v2 (test) ===")
for name, p in [("Qwen3", "adapted_l0/l0ab_qwen3vl_l0v2.jsonl"),
                ("Qwen2.5", "adapted_l0/l0ab_qwen25vl_l0v2.jsonl")]:
    rows = load(p)
    vals = []
    for arm in ("L0", "L1", "L2", "L3"):
        o, _ = inst(rows, "instance_object", arm)
        m, _ = inst(rows, "instance_material", arm)
        vals.append(f"{arm}=obj{o:.3f}/fine{m:.3f}")
    out(f"{name}: " + " ".join(vals))

out("\n=== SEED2 v2 ===")
s2 = load("adapted/adapted_qwen3vl_seed2v2.jsonl")
vals = []
for arm in ("L0", "L1"):
    o, _ = inst(s2, "instance_object", arm)
    m, _ = inst(s2, "instance_material", arm)
    vals.append(f"{arm}=obj{o:.3f}/fine{m:.3f}")
a, n = scene(s2, "cdw_seg", "F3_dominance", "test")
vals.append(f"F3cdwSeg={a:.3f}(n{n})")
out("Qwen3 seed2: " + " ".join(vals))

out("\n=== SEGMENTATION BASELINE (rebuilt keys) ===")
sb = load("seg_baseline/seg_baseline.jsonl")
for ds in ("codd", "cdw_seg"):
    cells = []
    for f in FAMS:
        sel = [r for r in sb if r.get("dataset") == ds and r["family"] == f]
        if ds == "codd":
            sel = [r for r in sel if str(r.get("split", "")).lower() in TEST]
        if sel:
            cells.append(f"{f[:2]}={sum(r['correct'] for r in sel)/len(sel):.3f}(n{len(sel)})")
    out(f"{ds}: " + " ".join(cells))

out("\n=== API MODELS v2 (end-to-end; F2/F3 rebuilt keys) ===")
for name, p in [("Kimi-k3", "api_eval/api_kimi_v2.jsonl"),
                ("GLM-5.3", "api_eval/api_glm_v2.jsonl")]:
    rows = load(p)
    out(f"{name}: {len(rows)} rows")
    for ds in ("codd", "cdw_seg"):
        cells = [f"{f[:2]}={scene(rows, ds, f)[0]:.3f}(n{scene(rows, ds, f)[1]})" for f in FAMS]
        out(f"  {ds}: " + " ".join(cells))
    for fam in ("instance_object", "instance_material"):
        vals = [f"{a}={inst(rows, fam, a)[0]:.3f}" for a in ("L0", "L1")]
        out(f"  {fam}: " + " ".join(vals))

out("\n=== HUMAN v3 (n=15 per cell) ===")
out("F1 0.733/0.733 (codd/CDW-Seg) | F4 0.800/0.800 | "
    "F2 0.733/0.467 | F3 0.400/0.600")

(R / "v2_number_sheet.txt").write_text("\n".join(LINES), encoding="utf-8")
print("\nWROTE", R / "v2_number_sheet.txt")
