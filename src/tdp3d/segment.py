"""Lesion segmentation (classical): blur -> Otsu -> morphology -> best contour."""
import cv2
import numpy as np


def segment_lesion(img_bgr):
    """Return a binary uint8 mask (0/1) of the lesion."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), max(h, w) / 100.0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, -1.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.005 * h * w:
            continue
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        d = np.hypot(cx - w / 2, cy - h / 2) / max(h, w)
        score = a * max(0.05, 1 - 2 * d)  # big and near the centre
        if score > best_score:
            best, best_score = c, score

    mask = np.zeros((h, w), np.uint8)
    if best is not None:
        cv2.drawContours(mask, [best], -1, 1, thickness=-1)
    frac = mask.mean()
    if frac < 0.01 or frac > 0.85:  # fallback: central ellipse
        mask[:] = 0
        cv2.ellipse(mask, (w // 2, h // 2), (w // 5, h // 5), 0, 0, 360, 1, -1)
    return mask
