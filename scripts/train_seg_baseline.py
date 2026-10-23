"""Trainable segmentation baseline: DeepLabV3-ResNet50 on the CODD train split.

Purpose (review P1-4.4): the benchmark's ground truth comes from segmentation
annotations, so a natural question is whether MLLMs are needed at all. This baseline trains a
standard supervised segmenter on the SAME CODD train images the MLLM SFT uses
and is evaluated on the same scene questions (eval_seg_baseline.py): in-domain
CODD test + zero-shot cross-domain CDW-Seg.

Classes: 10 CODD classes + background. Images resized 448x448, AdamW, cosine,
40 epochs, AMP on one GPU. Usage: python scripts/train_seg_baseline.py
"""
import sys, math, time, random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.cdw_vqa.codd import parse_xml, CLASSES as CODD_CLASSES

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

CODD = Path(r"datasets/codd/Construction and Demolition Waste Object Detection Dataset  (CODD)")
OUT = Path(r"models/seg_baseline")
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 448
EPOCHS = 40
random.seed(20261019)
torch.manual_seed(20261019)

CLS = sorted(CODD_CLASSES)
CIDX = {c: i + 1 for i, c in enumerate(CLS)}  # 0 = background


def load_split(split):
    """Benchmark train split for CODD = QA items with split in (train, training)."""
    import json
    qa = [json.loads(l) for l in open(BASE / "data" / "benchmark_v1_scene_qa.jsonl",
                                      encoding="utf-8")]
    want = {"train", "training"} if split == "train" else {"val", "validation"}
    ids = {r["image_id"] for r in qa if r["dataset"] == "codd" and r["split"] in want}
    xs = []
    for sp in ["training", "validation", "testing"]:
        for xp in sorted(CODD.joinpath(sp).glob("*.xml")):
            if xp.stem in ids:
                xs.append(xp)
    return xs


class SegData(Dataset):
    def __init__(self, xmls):
        self.xmls = xmls

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, i):
        ci = parse_xml(self.xmls[i])
        img = cv2.imdecode(np.fromfile(str(ci.path), dtype=np.uint8),
                           cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        m = np.zeros((ci.height, ci.width), np.uint8)
        for inst in ci.instances:  # later instances overwrite; fine for class maps
            pts = np.round(inst.polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [pts], CIDX[inst.name])
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = (img - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
              torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img, torch.from_numpy(m).long()


def main():
    dev = "cuda"
    try:
        from torchvision.models import ResNet50_Weights
        model = deeplabv3_resnet50(num_classes=len(CLS) + 1,
                                   weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
        print("backbone: ImageNet pretrained")
    except Exception as e:
        print("pretrained download failed (%s); random init" % type(e).__name__)
        model = deeplabv3_resnet50(num_classes=len(CLS) + 1)
    model = model.to(dev)

    tr = DataLoader(SegData(load_split("train")), batch_size=8, shuffle=True,
                    num_workers=0, drop_last=True)
    print("train images:", len(tr.dataset))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total = EPOCHS * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=total)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        run = 0.0
        for i, (x, y) in enumerate(tr):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                out = model(x)["out"]
                loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
        print(f"epoch {ep+1}/{EPOCHS} loss {run/len(tr):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), OUT / "codd_deeplab.pt")
    print("saved", OUT / "codd_deeplab.pt")


if __name__ == "__main__":
    main()
