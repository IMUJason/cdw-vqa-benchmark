"""CDW-Seg COCO parser: per-image class stats (area shares)."""
from __future__ import annotations
import json
from pathlib import Path

CAT_MAP = {  # cat_id -> (code, unified_class, material)
    1: ("BIN", "skip_bin", None),
    2: ("CB", "cardboard", "cardboard"),
    3: ("FD", "fill_dirt", "soil/fill dirt"),
    4: ("WT", "wood", "wood"),
    5: ("HP", "hard_plastic", "rigid plastic"),
    6: ("SP", "soft_plastic", "soft plastic/film"),
    7: ("ST", "steel", "steel/metal"),
    8: ("FB", "fabric", "fabric/textile"),
    9: ("CP", "concrete", "concrete"),
    10: ("PB", "plasterboard", "gypsum"),
}
NON_MINERAL = {"cardboard", "wood", "hard_plastic", "soft_plastic", "fabric"}


class CdwSeg:
    def __init__(self, root: str | Path):
        root = Path(root)
        d = json.load(open(root / "annotations.json", encoding="utf-8"))
        self.img_dir = root / "JPEGImages"
        self.images = {im["id"]: im for im in d["images"]}
        self.per_image = {}
        for a in d["annotations"]:
            code, cls, _ = CAT_MAP[a["category_id"]]
            self.per_image.setdefault(a["image_id"], []).append(
                dict(cls=cls, area=float(a.get("area", 0.0))))

    def stats(self, image_id: int) -> dict:
        """cls -> (n, area, share) excluding skip_bin from shares' denominator."""
        out = {}
        for a in self.per_image.get(image_id, []):
            n, ar = out.get(a["cls"], (0, 0.0))
            out[a["cls"]] = (n + 1, ar + a["area"])
        non_bin = {k: v for k, v in out.items() if k != "skip_bin"}
        total = sum(v[1] for v in non_bin.values()) or 1.0
        return {k: (v[0], v[1], v[1] / total) for k, v in non_bin.items()}

    def path(self, image_id: int) -> Path:
        return self.img_dir / Path(self.images[image_id]["file_name"]).name
