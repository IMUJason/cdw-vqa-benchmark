"""CODD VOC XML parsing: instances, classes, polygon masks."""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

CLASSES = ["brick", "concrete", "foam", "general_w", "gypsum_board",
           "pipes", "plastic", "stone", "tile", "wood"]


@dataclass
class Instance:
    name: str
    polygon: np.ndarray  # (N,2) float xy pixels


@dataclass
class CoddImage:
    image_id: str
    path: Path
    width: int
    height: int
    instances: list

    def mask_of(self, inst: Instance) -> np.ndarray:
        m = np.zeros((self.height, self.width), np.uint8)
        pts = np.round(inst.polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255)
        return m

    def union_mask(self, exclude: Instance | None = None) -> np.ndarray:
        m = np.zeros((self.height, self.width), np.uint8)
        for it in self.instances:
            if exclude is not None and it is exclude:
                continue
            pts = np.round(it.polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [pts], 255)
        return m


def parse_xml(xml_path: str | Path) -> CoddImage:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.findtext("width")) if size is not None else 0
    h = int(size.findtext("height")) if size is not None else 0
    img = cv2.imread(str(xml_path.with_suffix(".jpg")))
    if img is None:
        raise FileNotFoundError(xml_path.with_suffix(".jpg"))
    h, w = img.shape[:2]
    instances = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            continue
        poly = obj.find("polygon")
        if poly is None:
            continue
        xs, ys = {}, {}
        for el in poly.iter():
            m = re.fullmatch(r"([xy])(\d+)", el.tag or "")
            if m and el.text:
                (xs if m.group(1) == "x" else ys)[int(m.group(2))] = float(el.text)
        if not xs:  # fallback: LabelMe-style <pt><x/><y/>
            xs_list = [float(p.text) for p in poly.findall(".//x")]
            ys_list = [float(p.text) for p in poly.findall(".//y")]
        else:
            xs_list = [xs[k] for k in sorted(xs)]
            ys_list = [ys[k] for k in sorted(ys)]
        if len(xs_list) < 3 or len(xs_list) != len(ys_list):
            continue
        instances.append(Instance(name=name, polygon=np.stack([xs_list, ys_list], axis=1)))
    return CoddImage(image_id=xml_path.stem, path=xml_path.with_suffix(".jpg"),
                     width=w, height=h, instances=instances)


def class_stats(ci: CoddImage) -> dict:
    """Per-image class -> (n_instances, area_px, area_share)."""
    out = {}
    for it in ci.instances:
        a = float(cv2.contourArea(it.polygon.astype(np.float32)))
        n, aa = out.get(it.name, (0, 0.0))
        out[it.name] = (n + 1, aa + a)
    total = sum(v[1] for v in out.values()) or 1.0
    return {k: (v[0], v[1], v[1] / total) for k, v in out.items()}
