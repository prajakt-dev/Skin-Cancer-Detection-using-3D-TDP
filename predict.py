"""CLI prediction:  python predict.py path/to/image.jpg"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tdp3d import analyze, load_image_bytes  # noqa: E402

if len(sys.argv) < 2:
    raise SystemExit("usage: python predict.py <image>")
bundle = joblib.load("models/model.joblib")
res = analyze(load_image_bytes(Path(sys.argv[1]).read_bytes()), bundle)
print(f"Prediction : {res['label']}")
print(f"P(malignant): {res['probability_malignant']:.3f}")
for k, v in res["abcd"].items():
    print(f"  {k}: {v}")
