import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tdp3d import build_volume, extract_features, segment_lesion, tdp_codes  # noqa: E402


def _fake_lesion():
    import cv2
    img = np.full((160, 160, 3), (170, 190, 225), np.uint8)
    cv2.circle(img, (80, 80), 35, (40, 70, 110), -1)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def test_volume_shape():
    _, vol = build_volume(_fake_lesion(), 128)
    assert vol.shape == (8, 128, 128) and vol.dtype == np.float32


def test_codes_range():
    _, vol = build_volume(_fake_lesion(), 64)
    assert tdp_codes(vol).max() < 64


def test_segmentation_finds_lesion():
    m = segment_lesion(_fake_lesion())
    assert 0.1 < m.mean() < 0.4 and m[80, 80] == 1


def test_feature_vector_is_finite_and_stable_length():
    v, names, *_ = extract_features(_fake_lesion())
    assert len(v) == len(names) == 106 and np.isfinite(v).all()
