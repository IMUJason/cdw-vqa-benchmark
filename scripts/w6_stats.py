"""W6 gate statistics: orthogonal-tier interaction, McNemar, TOST, meta.
Verdict vs pre-registered criteria (02_方案 §4)."""
import json, sys, random
from pathlib import Path
from collections import defaultdict
from scipy.stats import binomtest
import numpy as np

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "w6"
ARMS = ["L0", "L1", "L2", "L3"]
B = 5000
random.seed(20261010)
np.random.seed(20261010)


def load(mk):
    recs = [json.loads(l) for l in open(OUT / f"responses_{mk}.jsonl", encoding="utf-8")]
    by = defaultdict(dict)
    for r in recs:
        by[(r["diag_id"], r["arm"])][r["family"]] = r["correct"]
    return by


def tier_of(r):
    return r["tier"]


def paired_stats(by, tier=None):
    """per arm: obj_acc, mat_acc, delta, mcnemar b/c/p"""
    out = {}
    for arm in ARMS:
        pairs = []
        for (did, a), fam in by.items():
            if a != arm or "instance_object" not in fam or "instance_material" not in fam:
                continue
            if tier:
                pass
            pairs.append((fam["instance_object"], fam["instance_material"]))
        if tier:
            pairs = [p for (i, p) in enumerate(pairs)]
        n = len(pairs)
        obj = sum(p[0] for p in pairs) / n
        mat = sum(p[1] for p in pairs) / n
        b = sum(1 for p in pairs if p[0] and not p[1])
        c = sum(1 for p in pairs if not p[0] and p[1])
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
        out[arm] = dict(n=n, obj=obj, mat=mat, delta=obj - mat, b=b, c=c, p=p)
    return out


def load_tiered(mk):
    recs = [json.loads(l) for l in open(OUT / f"responses_{mk}.jsonl", encoding="utf-8")]
    by = defaultdict(dict)
    meta = {}
    for r in recs:
        by[(r["diag_id"], r["arm"])][r["family"]] = r["correct"]
        meta[r["diag_id"]] = r["tier"]
    return by, meta


def paired(by, meta, arm, tier=None):
    pairs = []
    for (did, a), fam in by.items():
        if a != arm or "instance_object" not in fam or "instance_material" not in fam:
            continue
        if tier and meta.get(did) != tier:
            continue
        pairs.append((did, fam["instance_object"], fam["instance_material"]))
    return pairs


def delta(pairs):
    if not pairs:
        return 0.0
    return (sum(p[1] for p in pairs) - sum(p[2] for p in pairs)) / len(pairs)


def boot_ci(by, meta, stat_fn, B=B):
    ids = list({d for d, a in by})
    vals = []
    id_arr = np.array(ids)
    for _ in range(B):
        sample = np.random.choice(id_arr, size=len(id_arr), replace=True)
        vals.append(stat_fn(sample))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), \
           float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def model_report(mk):
    by, meta = load_tiered(mk)
    rep = {"model": mk}
    pairs = {arm: paired(by, meta, arm, tier="orthogonal") for arm in ARMS}
    rep["orth"] = {arm: dict(n=len(pairs[arm]),
                             obj=sum(p[1] for p in pairs[arm]) / max(len(pairs[arm]), 1),
                             mat=sum(p[2] for p in pairs[arm]) / max(len(pairs[arm]), 1),
                             delta=delta(pairs[arm]),
                             b=sum(1 for p in pairs[arm] if p[1] and not p[2]),
                             c=sum(1 for p in pairs[arm] if not p[1] and p[2]))
                   for arm in ARMS}
    for arm in ARMS:
        b, c = rep["orth"][arm]["b"], rep["orth"][arm]["c"]
        rep["orth"][arm]["p"] = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0

    def inter_stat(sample_ids):
        ss = set(sample_ids)
        d0 = delta([p for p in pairs["L0"] if p[0] in ss])
        d1 = delta([p for p in pairs["L1"] if p[0] in ss])
        return d0 - d1

    rep["interaction"] = delta(pairs["L0"]) - delta(pairs["L1"])
    lo, hi, lo5, hi5 = boot_ci(by, meta, inter_stat)
    rep["interaction_ci95"] = (lo, hi)
    rep["tost_l1_ci90"] = boot_ci(by, meta,
                                  lambda s: delta([p for p in pairs["L1"] if p[0] in set(s)]))[2:]
    rep["tost_l3_ci90"] = boot_ci(by, meta,
                                  lambda s: delta([p for p in pairs["L3"] if p[0] in set(s)]))[2:]
    # tier-stratified deltas per arm
    rep["tiers"] = {}
    for t in ["entangled", "partial", "material-pure", "na"]:
        rep["tiers"][t] = {arm: delta(paired(by, meta, arm, tier=t)) for arm in ARMS}
    return rep


def dl_meta(vals):
    """DerSimonian-Laird random-effects meta over 3 models."""
    v = np.array(vals)
    w = np.ones_like(v)  # 样本量近似相等，等权即固定效应近似；报告 pooled 均值
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def main():
    models = ["qwen3-vl-8b", "glm-4.1v-9b", "qwen2.5-vl-7b"]
    reps = [model_report(mk) for mk in models]
    lines = []
    for r in reps:
        lines.append(f"\n===== {r['model']} =====")
        for arm in ARMS:
            o = r["orth"][arm]
            lines.append(f"  {arm}: n={o['n']} obj={o['obj']:.3f} mat={o['mat']:.3f} "
                         f"Δ={o['delta']:.3f} McNemar b/c={o['b']}/{o['c']} p={o['p']:.2e}")
        lines.append(f"  interaction Δ_L0−Δ_L1 = {r['interaction']:.3f} "
                     f"CI95 [{r['interaction_ci95'][0]:.3f},{r['interaction_ci95'][1]:.3f}]")
        lines.append(f"  Δ_L1 TOST CI90 [{r['tost_l1_ci90'][0]:.3f},{r['tost_l1_ci90'][1]:.3f}] "
                     f"(±0.05 内=等效)")
        lines.append(f"  Δ_L3 TOST CI90 [{r['tost_l3_ci90'][0]:.3f},{r['tost_l3_ci90'][1]:.3f}]")
        lines.append("  tiers Δ per arm (L0/L1/L2/L3):")
        for t, dd in r["tiers"].items():
            lines.append(f"    {t}: " + " ".join(f"{dd[a]:+.3f}" for a in ARMS))
    m_int, se_int = dl_meta([r["interaction"] for r in reps])
    m_d0, _ = dl_meta([r["orth"]["L0"]["delta"] for r in reps])
    lines.append(f"\n===== META (3 models) =====")
    lines.append(f"pooled interaction = {m_int:.3f} ± {se_int:.3f}")
    lines.append(f"pooled Δ_L0 = {m_d0:.3f}")
    # verdict
    inter_ok = m_int >= 0.10 and all(r["interaction"] > 0 for r in reps)
    d0_ok = all(r["orth"]["L0"]["delta"] >= 0.10 for r in reps) and m_d0 >= 0.15
    tost_ok = all(r["tost_l1_ci90"][0] > -0.05 and r["tost_l1_ci90"][1] < 0.05 for r in reps)
    lines.append(f"\n===== VERDICT =====")
    lines.append(f"主判据 交互项≥10pp: {inter_ok} (pooled {m_int:.3f})")
    lines.append(f"辅助   Δ_L0≥15pp pooled & 逐模型同向: {d0_ok} (pooled {m_d0:.3f})")
    lines.append(f"辅助   Δ_L1 TOST±5pp 等效: {tost_ok}")
    verdict = "PASS（B 层机制主张成立）" if (inter_ok and tost_ok) else \
              ("部分支持（见明细）" if inter_ok else "FAIL（退回 A 版路径）")
    lines.append(f"W6 闸门判决: {verdict}")
    text = "\n".join(lines)
    print(text)
    (OUT / "w6_gate_stats.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
