# Skin Cancer Detection using 3D-TDP

A complete, runnable project that classifies dermoscopic skin-lesion images as **benign** or
**malignant (suspicious)** using a **3D Texture-Descriptor Pattern (3D-TDP)** pipeline, with a web
interface, evaluation reports and Docker support.

> **Interpretation of "3D-TDP" used here:** *3D Texture-Descriptor Pattern*. Each 2D lesion image is
> lifted into a 3D volume of 8 complementary depth slices; a 6-neighbour 3D local pattern code is
> computed for every voxel and its histogram becomes the descriptor. If your syllabus defines 3D-TDP
> differently, only `src/tdp3d/features.py` needs to change - the rest (data, training, web app,
> Docker) stays the same.

---
## 1. How it works (explain this to the examiner)

```
Image -> resize 128x128 -> lesion segmentation (Gaussian + Otsu + morphology)
      -> 3D volume V(z,y,x): z = 8 slices
           [Gray, CLAHE, Blur, Sobel edge, Laplacian, Saturation, Hue, Local-Std]
      -> 3D-TDP: every voxel compared with its 6 neighbours (+-x, +-y, +-z)
           => 6-bit code (0..63) => normalised histogram inside lesion (64 features)
      -> + per-slice lesion statistics & lesion-vs-skin contrast (24)
      -> + ABCD features: Asymmetry, Border (compactness, solidity, edge strength),
           Colour (mean/std/diff RGB, number of colours), Diameter (area, aspect)  (18)
      => 106 features -> Random Forest / Gradient Boosting / SVM (best chosen by 5-fold CV)
      -> probability of malignancy -> web UI
```

**Why 3D?** Stacking different filtered views along a depth axis lets the descriptor capture
*cross-view* relationships (e.g. a pixel that is dark in Gray but strong in Edge and Hue), which a
plain 2D texture descriptor cannot. The ABCD features follow the clinical dermatology rule.

## 2. Project structure

```
skin-cancer-3d-tdp/
├── app.py                      Flask web app (UI + /api/predict + /health)
├── train.py                    feature extraction, model selection, evaluation, reports
├── predict.py                  command-line prediction
├── train_cnn.py                OPTIONAL PyTorch 3D-CNN variant
├── src/tdp3d/                  segment.py  volume.py  features.py (3D-TDP)  pipeline.py
├── scripts/
│   ├── generate_synthetic_data.py   creates the offline demo dataset
│   ├── prepare_ham10000.py          converts the REAL HAM10000 dataset
│   └── bootstrap.py                 Docker entrypoint
├── data/images/{benign,malignant}/  dataset (500 demo images included)
├── data/samples/                    8 unseen images for the live demo
├── models/model.joblib              trained model (included)
├── reports/                         metrics.json, confusion matrix, ROC, feature importance
├── templates/ static/               web UI
├── tests/test_pipeline.py
├── Dockerfile  docker-compose.yml  requirements.txt
└── .vscode/                         launch + settings for VS Code
```

## 3. Run in VS Code (without Docker)

Prerequisites: **Python 3.10 - 3.12** and VS Code with the *Python* extension.

1. Unzip the project, then in VS Code: **File -> Open Folder -> skin-cancer-3d-tdp**.
2. Open the terminal (**Ctrl + `**) and create a virtual environment:
   - Windows (PowerShell): `python -m venv .venv` then `.venv\Scripts\Activate.ps1`
     (if blocked: `Set-ExecutionPolicy -Scope Process Bypass`)
   - macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`
3. Select the interpreter: **Ctrl+Shift+P -> "Python: Select Interpreter" -> .venv**.
4. Install dependencies: `pip install -r requirements.txt`
5. *(Dataset is already included. To regenerate: `python scripts/generate_synthetic_data.py --force`)*
6. Train: `python train.py`  (about 10 s; prints metrics, writes `models/` and `reports/`)
7. Start the app: `python app.py` and open **http://127.0.0.1:5000**
8. In the browser click a demo sample (or upload any image) to see the prediction, segmentation,
   ABCD values and the 3D-TDP volume slices.

Shortcuts: **Run and Debug** panel (F5) has ready configs: *Generate demo dataset*, *Train model*,
*Run web app*. Tests: `pip install pytest` then `pytest tests`. CLI: `python predict.py data/samples/sample_malignant_1.jpg`.

## 4. Run with Docker

Install Docker Desktop, then in the project folder:

```bash
docker compose up --build          # first run builds the image (~2 min)
# open http://localhost:5000
docker compose --profile train run --rm trainer     # (optional) retrain inside Docker
docker compose down                                  # stop
```
Plain Docker alternative:
```bash
docker build -t skin-cancer-3d-tdp .
docker run -p 5000:5000 -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" skin-cancer-3d-tdp
```
The container auto-generates data and trains a model if they are missing, then serves with gunicorn.
Health check: `http://localhost:5000/health`.

## 5. Using the REAL dataset (recommended for the final report)

The included images are **synthetic** (generated so the project runs offline). For real clinical
images use **HAM10000** (10,015 dermoscopic images; free):
- Kaggle: search "Skin Cancer MNIST: HAM10000" - or - Harvard Dataverse / ISIC Archive.

```bash
python scripts/prepare_ham10000.py --src path/to/ham10000 --max-per-class 1500
python train.py
```
Mapping: malignant = `mel, bcc, akiec`; benign = `nv, bkl, df, vasc`. The app and Docker use the new
model automatically (restart the app).  Real-data accuracy will be lower than on the synthetic set -
report it honestly.

## 6. Results on the included demo data (100 held-out images)

Gradient Boosting selected by CV: accuracy about 89 %, sensitivity 92 %, specificity 86 %, ROC-AUC 0.96
(see `reports/metrics.json` and the PNG plots; exact numbers vary slightly per run/platform).
These numbers are for **synthetic** data and must not be presented as clinical performance.

## 7. Likely viva questions

- **What is 3D-TDP?** A 3D local pattern histogram computed on a stack of filtered views of the lesion.
- **Why not only a CNN?** Small, explainable, fast, no GPU; `train_cnn.py` shows the deep variant.
- **What are ABCD features?** Asymmetry, Border irregularity, Colour variation, Diameter - the clinical rule.
- **How did you avoid leakage?** Stratified train/test split; model chosen with CV on the training part only.
- **Which metric matters most?** Sensitivity (missing a cancer is worse than a false alarm); ROC-AUC for overall quality.
- **Limitations?** Synthetic demo data, binary task, no skin-tone diversity analysis, not clinically validated.
- **Future work:** real HAM10000/ISIC training, 3D total-body photography, Grad-CAM, multi-class (7 types).

## 8. Troubleshooting
- `ModuleNotFoundError` -> venv not active / `pip install -r requirements.txt` not run.
- Port 5000 busy (macOS AirPlay) -> change the port at the bottom of `app.py` (and in docker-compose).
- Docker on Windows: make sure Docker Desktop is running; use PowerShell and `${PWD}` instead of `$(pwd)`.

**Disclaimer:** educational prototype, not a medical device.
