"""Docker entrypoint: make sure data + model exist, then start gunicorn."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

images = ROOT / "data" / "images"
if not images.exists() or not any(images.rglob("*.jpg")) and not any(images.rglob("*.png")):
    print("[bootstrap] no dataset found -> generating synthetic demo data", flush=True)
    subprocess.check_call([sys.executable, "scripts/generate_synthetic_data.py"])
if not (ROOT / "data" / "samples").exists() or not any((ROOT / "data" / "samples").glob("*.jpg")):
    subprocess.check_call([sys.executable, "scripts/generate_synthetic_data.py", "--force"])
if not (ROOT / "models" / "model.joblib").exists():
    print("[bootstrap] no trained model -> training now", flush=True)
    subprocess.check_call([sys.executable, "train.py"])

print("[bootstrap] starting web server on :5000", flush=True)
os.execvp("gunicorn", ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "--timeout", "120", "app:app"])
