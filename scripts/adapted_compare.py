"""Before/after comparison: zero-shot vs LoRA-adapted on test splits."""
import json, sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test",
             "train": "train", "val": "val", "test": "test"}


def acc(recs, key_fn, key):
    sub = [r for r in recs if key_fn(r) == key]
    if not sub:
        return None
    return sum(r["correct"] for r in sub) / len(sub), len(sub)


def main(model="qwen3vl", zs_scene="qwen3-vl-8b", zs_inst="qwen3-vl-8b"):
    OUT = BASE / "results"
    ad = [json.loads(l) for l in open(OUT / "adapted" / f"adapted_{model}.jsonl", encoding="utf-8")]
    zs_s = [json.loads(l) for l in open(OUT / "scene_eval" / f"scene_{zs_scene}.jsonl", encoding="utf-8")]
    zs_s = [r for r in zs_s if SPLIT_MAP[r["split"]] == "test"]
    zs_i = [json.loads(l) for l in open(OUT / "w6" / f"responses_{zs_inst}.jsonl", encoding="utf-8")]
    zs_i = [r for r in zs_i if r["split"] == "test"]
    ad_s = [r for r in ad if "dataset" in r]
    ad_i = [r for r in ad if "dataset" not in r]

    lines = [f"===== {model} 零样本 vs LoRA 适配（test split）====="]
    lines.append("\n[场景] family: zs -> adapted (n)")
    for fam in ["F1_presence", "F2_extent", "F3_dominance", "F4_contamination"]:
        for ds in ["codd", "cdw_seg"]:
            z = acc(zs_s, lambda r: (r["family"], r["dataset"]), (fam, ds))
            a = acc(ad_s, lambda r: (r["family"], r["dataset"]), (fam, ds))
            if z and a:
                lines.append(f"  {fam:18s} {ds:8s} {z[0]:.3f} -> {a[0]:.3f} "
                             f"({'+' if a[0] >= z[0] else ''}{a[0] - z[0]:.3f}) (n={z[1]})")
    lines.append("\n[实例-pipes正交] arm: zs_obj->ad_obj | zs_mat->ad_mat")
    for arm in ["L0", "L1", "L2", "L3"]:
        zo = acc(zs_i, lambda r: (r["arm"], r["family"], r["tier"]), (arm, "instance_object", "orthogonal"))
        ao = acc(ad_i, lambda r: (r["arm"], r["family"], r["tier"]), (arm, "instance_object", "orthogonal"))
        zm = acc(zs_i, lambda r: (r["arm"], r["family"], r["tier"]), (arm, "instance_material", "orthogonal"))
        am = acc(ad_i, lambda r: (r["arm"], r["family"], r["tier"]), (arm, "instance_material", "orthogonal"))
        if zo and ao and zm and am:
            lines.append(f"  {arm}: obj {zo[0]:.3f}->{ao[0]:.3f} | mat {zm[0]:.3f}->{am[0]:.3f} (n={zo[1]})")
    lines.append("\n[实例-纠缠类] arm: zs->adapted (object)")
    for arm in ["L0", "L1", "L2", "L3"]:
        z = acc(zs_i, lambda r: (r["arm"], r["family"], r["tier"]), (arm, "instance_object", "entangled"))
        a = acc(ad_i, lambda r: (r["arm"], r["family"], r["tier"]), (arm, "instance_object", "entangled"))
        if z and a:
            lines.append(f"  {arm}: {z[0]:.3f}->{a[0]:.3f} (n={z[1]})")
    text = "\n".join(lines)
    print(text)
    (BASE / "results" / "adapted" / f"before_after_{model}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
