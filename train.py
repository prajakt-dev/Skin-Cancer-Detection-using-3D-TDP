"""Train + evaluate the 3D-TDP skin-cancer classifier.

    python train.py                 # uses data/images/{benign,malignant}
    python train.py --data path/to/images --test-size 0.2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tdp3d import SLICE_NAMES, extract_features  # noqa: E402

CLASSES = ["benign", "malignant"]


def _one(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    vec, names, *_ = extract_features(img)
    return vec, names


def load_dataset(root):
    files, labels = [], []
    for y, c in enumerate(CLASSES):
        for p in sorted((Path(root) / c).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                files.append(p)
                labels.append(y)
    if not files:
        raise SystemExit(f"No images found in {root}/benign and {root}/malignant. "
                         "Run: python scripts/generate_synthetic_data.py")
    print(f"[data] {len(files)} images  (benign={labels.count(0)}, malignant={labels.count(1)})")
    t = time.time()
    res = Parallel(n_jobs=-1)(delayed(_one)(p) for p in files)
    keep = [i for i, r in enumerate(res) if r is not None]
    X = np.stack([res[i][0] for i in keep])
    y = np.array([labels[i] for i in keep])
    names = res[keep[0]][1]
    print(f"[features] {X.shape[1]} 3D-TDP features per image, extracted in {time.time() - t:.1f}s")
    return X, y, names, [files[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/images")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    Path(a.models_dir).mkdir(exist_ok=True)
    Path(a.reports_dir).mkdir(exist_ok=True)

    X, y, names, files = load_dataset(a.data)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=a.test_size, stratify=y, random_state=a.seed)

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                               random_state=a.seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=a.seed),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=3, probability=True,
                                                       class_weight="balanced", random_state=a.seed)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=a.seed)
    cv_scores = {}
    print("\n[model selection] 5-fold CV ROC-AUC on the training split")
    for n, m in candidates.items():
        s = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        cv_scores[n] = (float(s.mean()), float(s.std()))
        print(f"   {n:<17} {s.mean():.4f} +- {s.std():.4f}")
    best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
    best = candidates[best_name].fit(Xtr, ytr)
    print(f"[best] {best_name}")

    proba = best.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yte, proba)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    metrics = {
        "model": best_name,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred)),
        "recall_sensitivity": float(recall_score(yte, pred)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(yte, pred)),
        "roc_auc": float(auc(fpr, tpr)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "cv_roc_auc": {k: {"mean": v[0], "std": v[1]} for k, v in cv_scores.items()},
        "n_features": int(X.shape[1]),
    }
    print("\n[test results]")
    print(classification_report(yte, pred, target_names=CLASSES, digits=3))
    print(f"ROC-AUC {metrics['roc_auc']:.4f} | sensitivity {metrics['recall_sensitivity']:.3f} "
          f"| specificity {metrics['specificity']:.3f}")

    joblib.dump({"model": best, "model_name": best_name, "feature_names": names,
                 "classes": CLASSES, "threshold": 0.5, "metrics": metrics},
                Path(a.models_dir) / "model.joblib")
    json.dump(metrics, open(Path(a.reports_dir) / "metrics.json", "w"), indent=2)

    # ---------------- figures
    rd = Path(a.reports_dir)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = np.array(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix ({best_name})")
    fig.tight_layout(); fig.savefig(rd / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(rd / "roc_curve.png", dpi=150); plt.close(fig)

    rf = RandomForestClassifier(n_estimators=300, random_state=a.seed, n_jobs=-1).fit(Xtr, ytr)
    imp = rf.feature_importances_
    top = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.barh([names[i] for i in top][::-1], imp[top][::-1], color="#3b7ddd")
    ax.set_title("Top-15 feature importances (3D-TDP + ABCD)")
    fig.tight_layout(); fig.savefig(rd / "feature_importance.png", dpi=150); plt.close(fig)

    # one example volume for the report
    _, _, _, vol, img = extract_features(cv2.imread(str(files[-1])))
    fig, axs = plt.subplots(2, 4, figsize=(9, 4.8))
    for d, ax in enumerate(axs.ravel()):
        ax.imshow(vol[d], cmap="viridis"); ax.set_title(f"z{d}: {SLICE_NAMES[d]}", fontsize=9); ax.axis("off")
    fig.suptitle("3D-TDP volume of one lesion (8 depth slices)")
    fig.tight_layout(); fig.savefig(rd / "tdp_volume_example.png", dpi=150); plt.close(fig)
    print(f"\n[saved] models/model.joblib and reports/*.png|json")


if __name__ == "__main__":
    main()
