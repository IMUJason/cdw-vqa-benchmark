"""Build the 120-item scene survey package (F1-F4 x CODD/CDW-Seg x 15).

Items are the same (dataset, image_id, family) set as the earlier single-annotator
scene baseline, rebuilt from the released corpus so that choice order and keys are
the released ones. Truth stays in the local project (`key_local.json` in the build
directory) and is NOT copied into the served package.

Output: deploy_scene_survey/items/items.json  (no answer fields)
        deploy_scene_survey/items/img/*.jpg    (1280-px cache)
"""
import json
import re
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
BANK = BASE / "data" / "human_baseline_items.json"
CORPUS = BASE / "data" / "benchmark_v1_scene_qa.jsonl"
CACHE = Path(r"datasets/plan3_scene_cache")
OUT = BASE / "deploy_scene_survey"
FAMS = {"F1_presence", "F2_extent", "F3_dominance", "F4_contamination"}

ZH_CLASS = {
    "brick": "砖块", "concrete": "混凝土", "tile": "瓷砖", "stone": "石块",
    "wood": "木材", "gypsum board": "石膏板", "pipes": "管件", "plastic": "塑料",
    "plastics": "塑料", "foam": "泡沫", "general mixed": "一般混杂废料",
    "general mixed waste": "一般混杂废料", "cardboard": "纸板", "fill dirt": "填土",
    "hard plastic": "硬质塑料", "soft plastic": "软质塑料", "steel": "钢材",
    "fabric": "织物", "plasterboard": "石膏板", "timber": "木材",
}
FAM_ZH = {
    "F2_extent": "图中可以看到几类不同的废料材质？（数类别数，不数数量）",
    "F3_dominance": "这堆废料中，可见面积占比最大的材质是哪种？",
    "F4_contamination": "这堆废料中是否含有非矿物类材质（如木材、塑料、织物、纸板）？",
}


def zh_class(name):
    return ZH_CLASS.get(str(name).strip().lower(), str(name))


def zh_question(r):
    q = r["question"]
    fam = r["family"]
    if fam == "F1_presence":
        m = re.search(r"Is there any ([a-z_ ]+) visible", q)
        cls = zh_class(m.group(1)) if m else "该材质"
        return f"图中这堆废料里能看到「{cls}」吗？"
    return FAM_ZH.get(fam, q)


def main():
    bank = [x for x in json.load(open(BANK, encoding="utf-8")) if x.get("family") in FAMS]
    keys = {(x["dataset"], x["image_id"], x["family"]) for x in bank}
    assert len(keys) == 120, len(keys)
    corpus = {(r["dataset"], r["image_id"], r["family"]): r
              for r in (json.loads(l) for l in open(CORPUS, encoding="utf-8"))}

    imgdir = OUT / "items" / "img"
    if OUT.exists():
        shutil.rmtree(OUT)
    imgdir.mkdir(parents=True)

    items, key = [], {}
    for i, (ds, iid, fam) in enumerate(sorted(keys, key=lambda k: (k[2], k[0], k[1]))):
        r = corpus[(ds, iid, fam)]
        src = CACHE / (Path(r["image_path"]).stem + ".jpg")
        assert src.exists(), src
        fname = f"{i:03d}.jpg"
        shutil.copyfile(src, imgdir / fname)
        uid = f"scene|{ds}|{iid}|{fam}"
        it = dict(uid=uid, image=fname, family=fam, qtype=r["qtype"],
                  zh=zh_question(r), en=r["question"], dataset=ds, split=r.get("split"))
        if r["qtype"] == "mcq":
            it["choices"] = {k: zh_class(v) for k, v in r["choices"].items()}
        items.append(it)
        key[uid] = dict(answer=r["answer"], answer_text=r.get("answer_text"),
                        qtype=r["qtype"], dataset=ds, family=fam)

    (OUT / "items").mkdir(exist_ok=True)
    (OUT / "items" / "items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "key_local.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    # leak check: no answer/answer_text keys in the served items
    blob = (OUT / "items" / "items.json").read_text(encoding="utf-8")
    for bad in ('"answer"', "answer_text"):
        assert bad not in blob, f"truth leaked into served items: {bad}"
    n_mcq = sum(1 for x in items if x["qtype"] == "mcq")
    print(f"items={len(items)} (mcq={n_mcq}, yesno={len(items)-n_mcq}) "
          f"images={len(list(imgdir.glob('*.jpg')))}")
    du = sum(f.stat().st_size for f in imgdir.glob("*.jpg")) / 1e6
    print(f"image payload {du:.1f} MB")


if __name__ == "__main__":
    main()
