# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:47:57 2026

@author: k4927
"""

# Data fold

import shutil
from pathlib import Path
import yaml

KFOLD_ROOT = Path("/content/drive/MyDrive/Grouped_K-fold_folds")
RF_UPLOAD_ROOT = Path("/content/drive/MyDrive/Roboflow_upload_folds")

RF_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

def copy_tree(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob("*"):
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)

for fold_dir in sorted(KFOLD_ROOT.glob("fold_*")):
    out_dir = RF_UPLOAD_ROOT / f"{fold_dir.name}_rf"

    with open(fold_dir / "data.yaml", "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    names = y["names"]

    copy_tree(fold_dir / "images" / "train", out_dir / "train" / "images")
    copy_tree(fold_dir / "labels" / "train", out_dir / "train" / "labels")
    copy_tree(fold_dir / "images" / "val",   out_dir / "valid" / "images")
    copy_tree(fold_dir / "labels" / "val",   out_dir / "valid" / "labels")

    rf_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "names": names
    }

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(rf_yaml, f, allow_unicode=True, sort_keys=False)


from pathlib import Path

RF_UPLOAD_ROOT = Path("/content/drive/MyDrive/Roboflow_upload_folds")

for d in sorted(RF_UPLOAD_ROOT.glob("*_rf")):
    train_img = len(list((d / "train" / "images").glob("*")))
    train_lbl = len(list((d / "train" / "labels").glob("*.txt")))
    valid_img = len(list((d / "valid" / "images").glob("*")))
    valid_lbl = len(list((d / "valid" / "labels").glob("*.txt")))
    print(d.name)
    print("  train images:", train_img, "train labels:", train_lbl)
    print("  valid images:", valid_img, "valid labels:", valid_lbl)
    print("  yaml exists:", (d / "data.yaml").exists())
    print("-" * 50)
