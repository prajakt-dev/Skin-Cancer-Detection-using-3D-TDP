"""Flask web demo for the 3D-TDP skin-cancer detector.  Run:  python app.py"""
import json
import sys
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from tdp3d import analyze, load_image_bytes  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
MODEL_PATH = ROOT / "models" / "model.joblib"
SAMPLES = ROOT / "data" / "samples"
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Run `python train.py` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.route("/")
def index():
    samples = sorted(p.name for p in SAMPLES.glob("*.jpg")) if SAMPLES.exists() else []
    metrics = None
    try:
        metrics = get_bundle()["metrics"]
    except Exception:
        pass
    return render_template("index.html", samples=samples, metrics=metrics)


@app.route("/samples/<path:name>")
def sample_file(name):
    return send_from_directory(SAMPLES, name)


@app.route("/reports/<path:name>")
def report_file(name):
    return send_from_directory(ROOT / "reports", name)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files and request.files["file"].filename:
            data = request.files["file"].read()
        elif request.form.get("sample"):
            data = (SAMPLES / Path(request.form["sample"]).name).read_bytes()
        else:
            return jsonify(error="No image provided."), 400
        return jsonify(analyze(load_image_bytes(data), get_bundle()))
    except Exception as e:  # keep the demo friendly
        return jsonify(error=str(e)), 400


@app.route("/health")
def health():
    return jsonify(status="ok", model_loaded=MODEL_PATH.exists())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
