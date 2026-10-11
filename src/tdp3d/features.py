"""3D-TDP (3D Texture-Descriptor Pattern) feature extraction.

Idea
----
1. The lesion image is lifted into a volume V(d, y, x) of D=8 complementary views.
2. Every voxel is compared with its six 3D neighbours (+-x, +-y, +-depth). The six
   comparisons form a 6-bit code (0..63) -> a "3D local pattern". The normalised
   histogram of these codes inside the lesion is the 3D-TDP texture descriptor.
3. It is combined with per-slice lesion statistics and clinical ABCD-style features
   (Asymmetry, Border, Colour, Diameter).
"""
import cv2
import numpy as np

from .segment import segment_lesion
from .volume import build_volume, SLICE_NAMES

WORK_SIZE = 128


def tdp_codes(vol, eps=1e-4):
    """6-neighbour 3D local pattern code for every voxel -> uint8 array (D,H,W)."""
    p = np.pad(vol, 1, mode="edge")
    c = vol + eps
    nbrs = [p[2:, 1:-1, 1:-1], p[:-2, 1:-1, 1:-1],
            p[1:-1, 2:, 1:-1], p[1:-1, :-2, 1:-1],
            p[1:-1, 1:-1, 2:], p[1:-1, 1:-1, :-2]]
    code = np.zeros(vol.shape, np.uint8)
    for k, n in enumerate(nbrs):
        code |= ((n >= c).astype(np.uint8) << k)
    return code


def _asymmetry(mask):
    ys, xs = np.where(mask > 0)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(bool)
    ious = []
    for flipped in (crop[:, ::-1], crop[::-1, :]):
        inter = np.logical_and(crop, flipped).sum()
        union = np.logical_or(crop, flipped).sum()
        ious.append(inter / max(union, 1))
    return 1.0 - float(np.mean(ious))


def extract_features(img_bgr):
    """Return (feature_vector, feature_names, mask, volume, resized_image)."""
    img = cv2.resize(img_bgr, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    mask = segment_lesion(img)
    img, vol = build_volume(img, WORK_SIZE)
    D = vol.shape[0]
    feats = {}

    # --- (1) 3D-TDP histogram inside the lesion
    codes = tdp_codes(vol)
    m3 = np.broadcast_to(mask.astype(bool), codes.shape)
    hist = np.bincount(codes[m3], minlength=64).astype(np.float64)
    hist /= max(hist.sum(), 1)
    for i, v in enumerate(hist):
        feats[f"tdp_bin_{i:02d}"] = v
    nz = hist[hist > 0]
    feats["tdp_entropy"] = float(-(nz * np.log2(nz)).sum())
    feats["tdp_uniformity"] = float((hist ** 2).sum())

    # --- (2) per-slice lesion statistics (inside vs. surrounding skin ring)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ring = (cv2.dilate(mask, k) - mask).astype(bool)
    inside = mask.astype(bool)
    if ring.sum() < 10:
        ring = ~inside
    for d in range(D):
        s = vol[d]
        feats[f"slice{d}_mean"] = float(s[inside].mean())
        feats[f"slice{d}_std"] = float(s[inside].std())
        feats[f"slice{d}_contrast"] = float(s[inside].mean() - s[ring].mean())

    # --- (3) ABCD-style shape / colour features
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area = max(cv2.contourArea(c), 1.0)
    per = cv2.arcLength(c, True)
    hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    feats["A_asymmetry"] = _asymmetry(mask)
    feats["B_compactness"] = float(per ** 2 / (4 * np.pi * area))
    feats["B_solidity"] = float(area / hull_area)
    feats["B_edge_strength"] = float(vol[3][(cv2.dilate(mask, k) - cv2.erode(mask, k)).astype(bool)].mean())
    feats["D_area_fraction"] = float(area / (WORK_SIZE ** 2))
    feats["D_aspect_ratio"] = float(max(rw, rh) / max(min(rw, rh), 1.0))

    rgb = img[..., ::-1].astype(np.float32)
    px = rgb[inside]
    skin = rgb[ring].mean(axis=0)
    for i, ch in enumerate("RGB"):
        feats[f"C_mean_{ch}"] = float(px[:, i].mean())
        feats[f"C_std_{ch}"] = float(px[:, i].std())
        feats[f"C_diff_{ch}"] = float(px[:, i].mean() - skin[i])
    q = (px // 64).astype(int)
    qid = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    cnt = np.bincount(qid, minlength=64) / len(qid)
    feats["C_num_colors"] = float((cnt > 0.03).sum())

    names = list(feats.keys())
    vec = np.array([feats[n] for n in names], dtype=np.float32)
    return vec, names, mask, vol, img
