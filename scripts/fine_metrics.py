"""Two-level (coarse/fine) material scoring + corrected interaction gate.

Levels (pipes, the orthogonal tier):
  fine   : PVC = {pvc, upvc, cpvc, vinyl, polyvinyl chloride};
           metal = {metal, metallic, steel, iron, copper, aluminum, aluminium,
                    brass, galvanised/galvanized}.
           A bare "plastic"/"plastics"/"polymer" is NOT fine-PVC.
  coarse : plastic-family = fine-PVC + {plastic, plastics, polymer,
           polypropylene, polyethylene}; metal as above.
Object scoring uses a symmetric alias set (pipe/pipes/piping/tube/tubes/
hose/hoses/conduit/tubing) -- disjoint from material sets.

Also computes:
  * corrected pre-registered interaction (Delta_L0 - Delta_L1) per model and
    pooled, with cluster bootstrap CI (cluster = instance, B=2000);
  * McNemar exact tests for zero-shot -> adapted gains on the test split;
  * error-composition shares for the material questions (wrong-material share
    of errors), parsed from results/w6/w6_typology.txt;
  * empirical conditional entropy of pipes material truth.

Writes results/fine_metrics/fine_metrics.txt (and .json).
"""
import json, math, os, random, re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
OUT = RES / "fine_metrics"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["L0", "L1", "L2", "L3"]

FINE = {
    "PVC": r"\b(pvc|upvc|cpvc|vinyl|polyvinyl\s+chloride)\b",
    "metal": r"\b(metal|metallic|steel|iron|copper|aluminum|aluminium|brass|galvanis?zed)\b",
}
COARSE_EXTRA_PVC = r"\b(plastic|plastics|polymer|polypropylene|polyethylene)\b"
OBJ = r"\b(pipe|pipes|piping|tube|tubes|hose|hoses|conduit|tubing)\b"


def clean(a: str) -> str:
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
    a = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</?answer>", "", a)
    a = re.sub(r"\*+", "", a)
    return a.strip().lower()


def obj_score(raw):
    return bool(re.search(OBJ, clean(raw)))


def mat_score(raw, truth):
    """truth in {'PVC','metal'} for pipes. Returns (fine, coarse)."""
    a = clean(raw)
    fine = bool(re.search(FINE[truth], a))
    if truth == "PVC":
        coarse = fine or bool(re.search(COARSE_EXTRA_PVC, a))
    else:
        coarse = fine
    return fine, coarse


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def pipes_rows(rows, split=None):
    return [r for r in rows if r.get("cls") == "pipes"
            and not r.get("needs_manual_label")
            and (split is None or r["split"] == split)]


def main():
    lines = []
    P = lines.append

    files = {
        "zs_qwen3": RES / "w6" / "responses_qwen3-vl-8b.jsonl",
        "zs_qwen25": RES / "w6" / "responses_qwen2.5-vl-7b.jsonl",
        "zs_glm41": RES / "w6" / "responses_glm-4.1v-9b.jsonl",
        "ad_qwen3": RES / "adapted" / "adapted_qwen3vl.jsonl",
        "ad_qwen25": RES / "adapted" / "adapted_qwen25vl.jsonl",
        "oracle_qwen3": RES / "oracle" / "oracle_boxed_qwen3vl.jsonl",
    }
    for k, p in files.items():
        assert p.exists(), p

    report = {}

    # ---------- 1. per-arm object / material(fine, coarse) on pipes ----------
    P("=" * 70)
    P("1. PIPES PER-ARM, two-level material scoring (all splits, n=348/arm-family)")
    P("=" * 70)
    for mk, p in files.items():
        rows = pipes_rows(load(p))
        rep = {}
        P(f"\n-- {mk}")
        for fam in ["instance_object", "instance_material"]:
            for arm in ARMS:
                sel = [r for r in rows if r["family"] == fam and r["arm"] == arm]
                if not sel:
                    continue
                if fam == "instance_object":
                    acc = sum(obj_score(r["raw"]) for r in sel) / len(sel)
                    P(f"  {arm} object          n={len(sel)} acc={acc:.3f}")
                    rep[(fam, arm)] = dict(n=len(sel), obj=acc)
                else:
                    fine = sum(mat_score(r["raw"], r["answer"])[0] for r in sel) / len(sel)
                    coarse = sum(mat_score(r["raw"], r["answer"])[1] for r in sel) / len(sel)
                    stored = sum(r["correct"] for r in sel) / len(sel)
                    P(f"  {arm} material fine   n={len(sel)} acc={fine:.3f}   "
                      f"coarse={coarse:.3f}   stored(coarse+alias)={stored:.3f}")
                    rep[(fam, arm)] = dict(n=len(sel), mat_fine=fine, mat_coarse=coarse,
                                           stored=stored)
        report[mk] = rep

    # test split variant for adapted comparisons
    P("\n" + "=" * 70)
    P("1b. same, TEST split only (n=189/188)")
    P("=" * 70)
    test_rep = {}
    for mk in ["zs_qwen3", "zs_qwen25", "zs_glm41", "ad_qwen3", "ad_qwen25"]:
        rows = pipes_rows(load(files[mk]), split="test")
        rep = {}
        P(f"\n-- {mk}")
        for fam in ["instance_object", "instance_material"]:
            for arm in ARMS:
                sel = [r for r in rows if r["family"] == fam and r["arm"] == arm]
                if not sel:
                    continue
                if fam == "instance_object":
                    acc = sum(obj_score(r["raw"]) for r in sel) / len(sel)
                    P(f"  {arm} object   n={len(sel)} acc={acc:.3f}")
                    rep[(fam, arm)] = dict(n=len(sel), obj=acc)
                else:
                    fine = sum(mat_score(r["raw"], r["answer"])[0] for r in sel) / len(sel)
                    coarse = sum(mat_score(r["raw"], r["answer"])[1] for r in sel) / len(sel)
                    P(f"  {arm} material n={len(sel)} fine={fine:.3f} coarse={coarse:.3f}")
                    rep[(fam, arm)] = dict(n=len(sel), mat_fine=fine, mat_coarse=coarse)
        test_rep[mk] = rep

    # ---------- 2. corrected interaction gate (fine scorer, all splits) ----------
    P("\n" + "=" * 70)
    P("2. CORRECTED INTERACTION GATE (fine scorer, orthogonal pipes, all splits)")
    P("   Delta = object - fine-material; interaction = Delta_L0 - Delta_L1")
    P("=" * 70)
    gate = {}
    for mk in ["zs_qwen3", "zs_qwen25", "zs_glm41"]:
        rows = pipes_rows(load(files[mk]))
        d = {}
        for arm in ARMS:
            so = [r for r in rows if r["family"] == "instance_object" and r["arm"] == arm]
            sm = [r for r in rows if r["family"] == "instance_material" and r["arm"] == arm]
            o = {r["diag_id"]: obj_score(r["raw"]) for r in so}
            m = {r["diag_id"]: mat_score(r["raw"], r["answer"])[0] for r in sm}
            ids = sorted(set(o) & set(m))
            d[arm] = (sum(o[i] for i in ids) / len(ids),
                      sum(m[i] for i in ids) / len(ids),
                      {i: (o[i], m[i]) for i in ids}, ids)
        inter = d["L0"][2 - 2 + 0] - d["L0"][1] - (d["L1"][0] - d["L1"][1])
        P(f"\n{mk}:")
        for arm in ARMS:
            P(f"  {arm}: obj={d[arm][0]:.3f} mat_fine={d[arm][1]:.3f} "
              f"Delta={d[arm][0] - d[arm][1]:+.3f}")
        P(f"  interaction (L0-L1) = {inter:+.3f}   [original stored scorer: see w6_gate_stats]")
        gate[mk] = dict(arms={a: dict(obj=d[a][0], mat=d[a][1],
                                      delta=d[a][0] - d[a][1]) for a in ARMS},
                        interaction=inter)
    pooled = sum(g["interaction"] for g in gate.values()) / 3
    P(f"\nPOOLED corrected interaction = {pooled:+.3f} (pre-registered threshold >= +0.10)")

    # cluster bootstrap CI for pooled interaction
    random.seed(20261019)
    zs = {mk: pipes_rows(load(files[mk])) for mk in gate}
    per_model_ids = {}
    for mk in gate:
        rows = zs[mk]
        ids = sorted(set(r["diag_id"] for r in rows))
        per_model_ids[mk] = ids
    ids_all = sorted(set.union(*[set(v) for v in per_model_ids.values()]))
    per_model_rows = {mk: {} for mk in gate}
    for mk in gate:
        for r in zs[mk]:
            per_model_rows[mk][(r["diag_id"], r["family"], r["arm"])] = r
    boots = []
    B = 2000
    for b in range(B):
        samp = [ids_all[random.randrange(len(ids_all))] for _ in range(len(ids_all))]
        tot = 0.0
        for mk in gate:
            def acc(arm, fam, fn):
                vals = []
                for i in samp:
                    r = per_model_rows[mk].get((i, fam, arm))
                    if r is None:
                        continue
                    vals.append(fn(r))
                return sum(vals) / len(vals) if vals else 0.0
            o0 = acc("L0", "instance_object", lambda r: obj_score(r["raw"]))
            m0 = acc("L0", "instance_material", lambda r: mat_score(r["raw"], r["answer"])[0])
            o1 = acc("L1", "instance_object", lambda r: obj_score(r["raw"]))
            m1 = acc("L1", "instance_material", lambda r: mat_score(r["raw"], r["answer"])[0])
            tot += (o0 - m0) - (o1 - m1)
        boots.append(tot / 3)
    boots.sort()
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B)]
    P(f"cluster bootstrap 95% CI (B={B}, cluster=instance): [{lo:+.3f}, {hi:+.3f}]")
    gate["pooled"] = dict(interaction=pooled, ci95=[lo, hi], B=B)
    report["gate_fine"] = gate

    # ---------- 3. McNemar zero-shot -> adapted (test split, paired) ----------
    P("\n" + "=" * 70)
    P("3. McNEMAR exact, zero-shot -> adapted (test split, paired by diag_id)")
    P("=" * 70)

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        from math import comb
        k = min(b, c)
        p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
        return min(1.0, p)

    pairs = [("zs_qwen3", "ad_qwen3", "Qwen3-VL-8B"),
             ("zs_qwen25", "ad_qwen25", "Qwen2.5-VL-7B")]
    mc = {}
    for zs_mk, ad_mk, label in pairs:
        zr = pipes_rows(load(files[zs_mk]), split="test")
        ar = pipes_rows(load(files[ad_mk]), split="test")
        P(f"\n-- {label}")
        for fam in ["instance_object", "instance_material"]:
            for arm in ARMS:
                z = {r["diag_id"]: (obj_score(r["raw"]) if fam == "instance_object"
                                    else mat_score(r["raw"], r["answer"])[0])
                     for r in zr if r["family"] == fam and r["arm"] == arm}
                a = {r["diag_id"]: (obj_score(r["raw"]) if fam == "instance_object"
                                    else mat_score(r["raw"], r["answer"])[0])
                     for r in ar if r["family"] == fam and r["arm"] == arm}
                ids = sorted(set(z) & set(a))
                b = sum(1 for i in ids if not z[i] and a[i])
                c = sum(1 for i in ids if z[i] and not a[i])
                p = mcnemar(b, c)
                key = f"{label}|{fam}|{arm}"
                mc[key] = dict(b=b, c=c, p=p,
                               zs=sum(z[i] for i in ids) / len(ids),
                               ad=sum(a[i] for i in ids) / len(ids))
                P(f"  {arm} {fam:17s} zs={mc[key]['zs']:.3f} ad={mc[key]['ad']:.3f} "
                  f"discordant {c}->{b} (worse->better) p={p:.2e}")
    report["mcnemar"] = mc

    # ---------- 4. error composition shares (material Qs) ----------
    P("\n" + "=" * 70)
    P("4. ERROR COMPOSITION, material questions (from w6_typology.txt)")
    P("   wrong-material share OF ERRORS = wrong_mat / (1 - correct)")
    P("=" * 70)
    typ = {}
    cur, in_mat = None, False
    for line in open(RES / "w6" / "w6_typology.txt", encoding="utf-8"):
        m = re.match(r"=====\s+(\S+)\s+\(pipes", line)
        if m:
            cur, in_mat = m.group(1), False
            typ[cur] = {}
            continue
        if cur and line.strip().startswith("[material-Q]"):
            in_mat = True
            continue
        if cur and line.strip().startswith("[object-Q]"):
            in_mat = False
            continue
        if cur and in_mat:
            arm_m = re.match(
                r"\s*(L\d): correct=([\d.]+) wrong_mat=([\d.]+) wrong_obj=([\d.]+) "
                r"\*\*hijack=([\d.]+)\*\* other=([\d.]+)", line)
            if arm_m:
                a, c0, wm, wo, hj, ot = arm_m.groups()
                typ[cur][a] = dict(correct=float(c0), wrong_mat=float(wm),
                                   hijack=float(hj))
                err = 1 - float(c0)
                share = float(wm) / err if err > 0 else float("nan")
                P(f"  {cur:16s} {a}: wrong_mat share of errors = {share:.2f} "
                  f"(hijack share {float(hj)/err:.2f})")
    report["typology_error_shares"] = {
        mk: {a: dict(wrong_mat_share=(v["wrong_mat"] / (1 - v["correct"])),
                     hijack_share=(v["hijack"] / (1 - v["correct"])))
             for a, v in d.items()} for mk, d in typ.items()}

    # ---------- 5. empirical conditional entropy of pipes truth ----------
    final = list(__import__("csv").DictReader(
        open(BASE / "data" / "pipes_material_final.csv", encoding="utf-8-sig")))
    n = sum(1 for r in final if r["material"] in ("PVC", "metal"))
    pp = sum(1 for r in final if r["material"] == "PVC") / n
    pm = 1 - pp
    H = -(pp * math.log2(pp) + pm * math.log2(pm))
    P("\n" + "=" * 70)
    P(f"5. EMPIRICAL H(material | pipes) = {H:.3f} bits "
      f"(PVC {pp:.3f} / metal {pm:.3f}, n={n})")
    P("   lexical tier: 'pipes' does not name its material (orthogonal);")
    P("   empirical entropy is LOW because the class prior is imbalanced.")
    report["pipes_empirical_entropy_bits"] = H

    def stringify(o):
        if isinstance(o, dict):
            return {str(k): stringify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [stringify(v) for v in o]
        return o

    txt = "\n".join(lines)
    (OUT / "fine_metrics.txt").write_text(txt, encoding="utf-8")
    (OUT / "fine_metrics.json").write_text(
        json.dumps(stringify(report), indent=1, default=str), encoding="utf-8")
    print(txt)
    print("\nWROTE", OUT / "fine_metrics.txt")


if __name__ == "__main__":
    main()
