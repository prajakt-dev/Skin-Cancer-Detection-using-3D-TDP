"""Generate a SYNTHETIC dermoscopy-like dataset (benign vs malignant) so the project runs
completely offline. Malignant-like lesions are statistically more asymmetric, have
irregular borders and several colours (ABCD rule) - with deliberate class overlap so the
results are realistic (not 100%).

For real clinical images use HAM10000 / ISIC (see README) via prepare_ham10000.py.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

SIZE = 160


def _smooth_noise(rng, size, sigma):
    n = rng.normal(0, 1, (size, size)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (np.abs(n).max() + 1e-6)


def make_image(label, rng):
    # atypia score: malignant tends high, benign tends low, with overlap
    atypia = rng.beta(6, 2) if label == 1 else rng.beta(2, 6)
    skin = np.array([rng.uniform(205, 235), rng.uniform(160, 190), rng.uniform(135, 170)])  # RGB
    bg = np.ones((SIZE, SIZE, 3), np.float32) * skin
    bg *= (1 + 0.06 * _smooth_noise(rng, SIZE, 30))[..., None]

    # irregular border via polar harmonics
    cx, cy = SIZE / 2 + rng.uniform(-8, 8), SIZE / 2 + rng.uniform(-8, 8)
    r0 = SIZE * rng.uniform(0.17, 0.25 + 0.12 * atypia)
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    rad = np.ones_like(theta)
    for k in range(2, 8):
        rad += rng.uniform(0, 0.03 + 0.14 * atypia) / (k ** 0.6) * np.cos(k * theta + rng.uniform(0, 6.28))
    elong = 1 + rng.uniform(0, 0.15 + 0.5 * atypia)
    ang = rng.uniform(0, np.pi)
    x = r0 * rad * np.cos(theta) * elong
    y = r0 * rad * np.sin(theta) / np.sqrt(elong)
    xr = cx + x * np.cos(ang) - y * np.sin(ang)
    yr = cy + x * np.sin(ang) + y * np.cos(ang)
    mask = np.zeros((SIZE, SIZE), np.float32)
    cv2.fillPoly(mask, [np.stack([xr, yr], 1).astype(np.int32)], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0 + 2.0 * (1 - atypia) * rng.uniform(0.3, 1))

    # lesion colours
    base = np.array([rng.uniform(95, 140), rng.uniform(55, 85), rng.uniform(35, 60)])
    les = np.ones((SIZE, SIZE, 3), np.float32) * base
    les *= (1 + 0.25 * _smooth_noise(rng, SIZE, 6))[..., None]
    palette = [(30, 20, 20), (165, 55, 55), (85, 100, 135), (205, 190, 180), (60, 35, 25)]
    for _ in range(int(round(atypia * rng.uniform(1, 6)))):
        col = np.array(palette[rng.integers(len(palette))], np.float32)
        px, py = cx + rng.normal(0, r0 * 0.5), cy + rng.normal(0, r0 * 0.5)
        blob = np.zeros((SIZE, SIZE), np.float32)
        cv2.circle(blob, (int(px), int(py)), int(rng.uniform(0.15, 0.4) * r0), 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 3)[..., None]
        les = les * (1 - blob) + col * blob

    img = bg * (1 - mask[..., None]) + les * mask[..., None]

    # thin hairs (distractors) and sensor noise
    for _ in range(rng.integers(0, 5)):
        p1 = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        p2 = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        cv2.line(img, p1, p2, (50, 35, 30), 1, cv2.LINE_AA)
    img += rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=250)
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--samples-out", default="data/samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing data")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.rglob("*.jpg")) and not a.force:
        print(f"[skip] {out} already has images (use --force to regenerate)")
        return
    if out.exists() and a.force:
        shutil.rmtree(out)
    rng = np.random.default_rng(a.seed)
    for label, name in ((0, "benign"), (1, "malignant")):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.n_per_class):
            cv2.imwrite(str(d / f"{name}_{i:04d}.jpg"), make_image(label, rng), [cv2.IMWRITE_JPEG_QUALITY, 92])
    # small gallery for the web demo (generated with a different seed -> unseen by training)
    s = Path(a.samples_out)
    s.mkdir(parents=True, exist_ok=True)
    rng2 = np.random.default_rng(a.seed + 999)
    for label, name in ((0, "benign"), (1, "malignant")):
        for i in range(4):
            cv2.imwrite(str(s / f"sample_{name}_{i + 1}.jpg"), make_image(label, rng2), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[ok] wrote {a.n_per_class}x2 images to {out} and 8 demo samples to {s}")


if __name__ == "__main__":
    main()
