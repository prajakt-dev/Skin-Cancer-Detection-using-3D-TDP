"""OPTIONAL deep-learning variant: a small 3D CNN (Conv3d) on the same 3D-TDP volumes.

    pip install -r requirements-cnn.txt
    python train_cnn.py --epochs 25

NOTE: this variant needs PyTorch and is NOT used by the web app / Docker image
(the main pipeline uses the lightweight 3D-TDP descriptor + classical classifier).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tdp3d import build_volume  # noqa: E402

SIZE = 64


class Tdp3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(),
                                 nn.MaxPool3d((1, 2, 2)))
        self.features = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):  # x: (B, 1, D=8, H, W)
        return self.head(self.features(x))


def load(root):
    X, y = [], []
    for label, c in enumerate(["benign", "malignant"]):
        for p in sorted((Path(root) / c).glob("*")):
            img = cv2.imread(str(p))
            if img is not None:
                X.append(build_volume(img, SIZE)[1][None]); y.append(label)
    return np.stack(X).astype(np.float32), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/images")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    X, y = load(a.data)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, opt = Tdp3DCNN().to(dev), None
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.softmax(net(torch.tensor(Xte).to(dev)), 1)[:, 1].cpu().numpy()
        print(f"epoch {ep + 1:02d}  test acc {accuracy_score(yte, p > .5):.3f}  auc {roc_auc_score(yte, p):.3f}")
    Path("models").mkdir(exist_ok=True)
    torch.save(net.state_dict(), "models/tdp3d_cnn.pt")


if __name__ == "__main__":
    main()
