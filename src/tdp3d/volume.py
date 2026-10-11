"""Build the 3D-TDP volume: a lesion image becomes a (D, H, W) tensor whose depth axis
holds D complementary 'views' (intensity, contrast, scale, edge, texture, colour)."""
import cv2
import numpy as np

SLICE_NAMES = ["Gray", "CLAHE", "Blur(s=2)", "Edge(Sobel)", "Laplacian",
               "Saturation", "Hue", "Local-Std"]


def _norm(x):
    return np.clip(x / (np.percentile(x, 99) + 1e-6), 0, 1).astype(np.float32)


def build_volume(img_bgr, size=128):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gf = gray.astype(np.float32)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gf, (0, 0), 2)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gf, (0, 0), 1), cv2.CV_32F))
    mu = cv2.blur(gf, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(gf * gf, (7, 7)) - mu ** 2, 0))
    vol = np.stack([
        gf / 255.0, clahe / 255.0, blur / 255.0,
        _norm(np.hypot(gx, gy)), _norm(lap),
        hsv[..., 1] / 255.0, hsv[..., 0] / 179.0, _norm(std),
    ]).astype(np.float32)
    return img, vol
