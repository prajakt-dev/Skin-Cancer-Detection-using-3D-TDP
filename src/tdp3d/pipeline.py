"""High-level helpers used by the web app and CLI."""
import base64

import cv2
import numpy as np

from .features import extract_features
from .volume import SLICE_NAMES


def load_image_bytes(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a JPG/PNG file.")
    return img


def _b64_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _montage(vol):
    tiles = []
    for d, name in enumerate(SLICE_NAMES):
        t = cv2.applyColorMap((vol[d] * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        t = cv2.resize(t, (160, 160), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(t, (0, 0), (160, 18), (0, 0, 0), -1)
        cv2.putText(t, f"z{d}: {name}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    return np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])


def analyze(img_bgr, bundle):
    """Run the full 3D-TDP pipeline on one image and return a JSON-friendly dict."""
    vec, names, mask, vol, img = extract_features(img_bgr)
    model = bundle["model"]
    proba = float(model.predict_proba(vec.reshape(1, -1))[0][1])
    thr = bundle.get("threshold", 0.5)
    label = "Malignant (suspicious)" if proba >= thr else "Benign (low suspicion)"

    overlay = cv2.resize(img, (384, 384), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 120), 2)

    f = dict(zip(names, vec.tolist()))
    abcd = {
        "Asymmetry (0=symmetric)": round(f["A_asymmetry"], 3),
        "Border irregularity (1=circle)": round(f["B_compactness"], 3),
        "Colours detected": int(f["C_num_colors"]),
        "Lesion size (% of image)": round(100 * f["D_area_fraction"], 1),
        "3D-TDP entropy (bits)": round(f["tdp_entropy"], 3),
    }
    return {
        "probability_malignant": round(proba, 4),
        "label": label,
        "is_malignant": bool(proba >= thr),
        "abcd": abcd,
        "overlay": _b64_png(overlay),
        "volume_montage": _b64_png(_montage(vol)),
        "model_name": bundle.get("model_name", "model"),
    }
