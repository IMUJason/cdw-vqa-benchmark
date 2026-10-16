"""Four-arm image generation (L0-L3) for W6 condition-contrast design."""
from __future__ import annotations
import numpy as np
import cv2

CANVAS = (896, 896)  # square canvas for L3 (w,h)


def arm_l0(img: np.ndarray) -> np.ndarray:
    """L0 in-situ: untouched."""
    return img.copy()


def arm_l1_inpaint(img: np.ndarray, target_mask: np.ndarray,
                   dilate: int = 7, band: int = 15) -> np.ndarray:
    """L1 main control: same canvas, context removed; target pixels/scale/
    position fully preserved. Fast path: mean-fill + inpaint only a thin
    boundary band (full-image Telea inpaint on 2MP was ~8s/instance)."""
    off = target_mask == 0
    out = img.copy()
    mean_color = img[off].mean(axis=0).astype(np.uint8) if off.any() \
        else np.array([128, 128, 128], np.uint8)
    out[off] = mean_color
    k = np.ones((band, band), np.uint8)
    dil = cv2.dilate(target_mask, k)
    bandmask = cv2.bitwise_and(cv2.bitwise_xor(dil, target_mask),
                               cv2.bitwise_not(target_mask))
    if bandmask.sum() > 0:
        soft = cv2.inpaint(img, cv2.dilate(bandmask, np.ones((5, 5), np.uint8)),
                           7, cv2.INPAINT_TELEA)
        out[bandmask > 0] = soft[bandmask > 0]
    return out


def arm_l1_fill(img: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """L1b variant: background filled with dataset-mean neutral color."""
    out = img.copy()
    mean_color = tuple(int(c) for c in img[target_mask == 0].mean(axis=0)) \
        if (target_mask == 0).any() else (128, 128, 128)
    out[target_mask == 0] = mean_color
    return out


def arm_l2_blur(img: np.ndarray, target_mask: np.ndarray,
                k: int = 51) -> np.ndarray:
    """L2: background Gaussian-blurred, target sharp."""
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    out = blurred.copy()
    out[target_mask > 0] = img[target_mask > 0]
    return out


def arm_l3_crop(img: np.ndarray, target_mask: np.ndarray,
                pad_frac: float = 0.08, canvas: tuple = CANVAS) -> np.ndarray:
    """L3 deployment-translation: tight crop of target bbox, then downscaled
    so the target's effective pixel count matches its L0 footprint (no
    resolution inflation), pasted centered on neutral canvas."""
    ys, xs = np.where(target_mask > 0)
    if len(xs) == 0:
        return cv2.resize(img, canvas)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(img.shape[1] - 1, x1 + px), min(img.shape[0] - 1, y1 + py)
    crop = img[y0:y1 + 1, x0:x1 + 1]
    m_crop = target_mask[y0:y1 + 1, x0:x1 + 1]
    n_target_l0 = int((target_mask > 0).sum())
    # choose crop scale so target pixels in output == n_target_l0
    ch, cw = canvas
    scale_full = min(cw / crop.shape[1], ch / crop.shape[0])
    # target area would scale by s^2; solve s so area matches, capped by canvas
    frac = n_target_l0 / (crop.shape[0] * crop.shape[1])
    s_area = (n_target_l0 / max(frac, 1e-9)) ** 0.5 / max(crop.shape[0], crop.shape[1])
    s = min(scale_full, max(s_area, 1e-3))
    nw, nh = max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    m_res = cv2.resize(m_crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.full((ch, cw, 3), 128, np.uint8)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    out[oy:oy + nh, ox:ox + nw][m_res > 0] = resized[m_res > 0]
    return out


def contact_sheet(panels: list, labels: list, cell: int = 448) -> np.ndarray:
    """panels: list of BGR images -> single row sheet resized to cell."""
    row = []
    for im, lb in zip(panels, labels):
        r = cv2.resize(im, (cell, cell))
        cv2.putText(r, lb, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        row.append(r)
    return np.concatenate(row, axis=1)
