"""Convert the REAL HAM10000 dataset (Kaggle / ISIC / Harvard Dataverse) to this project's
folder layout:  data/images/benign/*.jpg   data/images/malignant/*.jpg

Usage:
  python scripts/prepare_ham10000.py --src path/to/ham10000 --max-per-class 1500

--src must contain HAM10000_metadata.csv and the image folders
(e.g. HAM10000_images_part_1 / HAM10000_images_part_2 or an 'images' folder).
Malignant = mel, bcc, akiec      Benign = nv, bkl, df, vasc
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

MALIGNANT = {"mel", "bcc", "akiec"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    meta = next(src.rglob("HAM10000_metadata.csv"), None)
    if meta is None:
        raise SystemExit("HAM10000_metadata.csv not found under --src")
    index = {p.stem: p for p in src.rglob("*.jpg")}
    groups = {"benign": [], "malignant": []}
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            p = index.get(row["image_id"])
            if p is not None:
                groups["malignant" if row["dx"] in MALIGNANT else "benign"].append(p)

    random.seed(a.seed)
    if out.exists():
        shutil.rmtree(out)
    for name, files in groups.items():
        random.shuffle(files)
        (out / name).mkdir(parents=True, exist_ok=True)
        for p in files[: a.max_per_class]:
            shutil.copy(p, out / name / p.name)
        print(f"{name}: {min(len(files), a.max_per_class)} images (of {len(files)} available)")
    print("Done. Now run:  python train.py")


if __name__ == "__main__":
    main()
