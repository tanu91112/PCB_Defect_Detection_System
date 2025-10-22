"""
GOAL: Build labeled ROI dataset from DeepPCB annotations
- Parse *_not/*.txt label files with bounding boxes and class IDs
- Crop defect regions from test images using ground truth boxes
- Split dataset into train/val/test with stratified sampling
- Organize by class folders for PyTorch ImageFolder loading
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random
import shutil
import cv2


DEFECT_CLASSES: Dict[int, str] = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "pinhole",
    6: "spurious copper",
}


# LABEL PARSING: Read bounding boxes and class IDs from annotation file
def read_labels(label_path: Path) -> List[Tuple[int, int, int, int, int]]:
    items: List[Tuple[int, int, int, int, int]] = []
    if not label_path.exists():
        return items
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2, cls = map(int, parts[:5])
            items.append((x1, y1, x2, y2, cls))
    return items


# GROUP DISCOVERY: Find all numeric group directories in dataset
def collect_groups(data_root: Path) -> List[Path]:
    groups = []
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        # expect subfolder like groupXXXXX containing XXXXX and XXXXX_not
        for inner in p.iterdir():
            if inner.is_dir():
                groups.append(inner)
        # we will filter later to only include numeric folders
    # Only keep numeric-named folders (e.g., 12000, 00041, etc.)
    groups = [g for g in groups if g.name.isdigit()]
    return groups


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ROI CROPPING: Extract and save defect regions from test images
def crop_and_save_rois(
    test_img_path: Path,
    labels: List[Tuple[int, int, int, int, int]],
    out_dir: Path,
    prefix: str,
) -> int:
    img = cv2.imread(str(test_img_path))
    if img is None:
        return 0
    saved = 0
    for i, (x1, y1, x2, y2, cls) in enumerate(labels):
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(img.shape[1], x2)
        y2c = min(img.shape[0], y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = img[y1c:y2c, x1c:x2c]
        cls_name = DEFECT_CLASSES.get(cls, str(cls))
        cls_dir = out_dir / cls_name
        ensure_dir(cls_dir)
        out_path = cls_dir / f"{prefix}_{i:03d}.jpg"
        cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved += 1
    return saved


# STRATIFIED SPLITTING: Split dataset maintaining class distribution
def stratified_split(items: List[Tuple[str, str]], seed: int, ratios=(0.7, 0.15, 0.15)):
    random.Random(seed).shuffle(items)
    # group by class
    class_to_items: Dict[str, List[Tuple[str, str]]] = {}
    for path, cls in items:
        class_to_items.setdefault(cls, []).append((path, cls))
    train, val, test = [], [], []
    for cls, cls_items in class_to_items.items():
        n = len(cls_items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        cls_train = cls_items[:n_train]
        cls_val = cls_items[n_train:n_train + n_val]
        cls_test = cls_items[n_train + n_val:]
        train.extend(cls_train)
        val.extend(cls_val)
        test.extend(cls_test)
    return train, val, test


# MAIN DATASET BUILDER: Complete workflow to create labeled ROI dataset
def build_dataset(
    data_root: Path,
    out_root: Path,
    seed: int = 42,
) -> None:
    ensure_dir(out_root)
    tmp_all = out_root / "_all"
    if tmp_all.exists():
        shutil.rmtree(tmp_all)
    ensure_dir(tmp_all)

    total_saved = 0
    groups = collect_groups(data_root)
    for numeric_dir in groups:
        # look for paired files *_test.jpg and a label in sibling *_not folder
        for test_img in numeric_dir.glob("*_test.jpg"):
            stem_id = test_img.stem.split("_")[0]
            label_dir = numeric_dir.parent / f"{numeric_dir.name}_not"
            label_path = label_dir / f"{stem_id}.txt"
            labels = read_labels(label_path)
            saved = crop_and_save_rois(test_img, labels, tmp_all, prefix=stem_id)
            total_saved += saved

    # gather all saved images with labels as class from directory name
    all_items: List[Tuple[str, str]] = []
    for cls_dir in tmp_all.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for img_path in cls_dir.glob("*.jpg"):
            all_items.append((str(img_path), cls_name))

    train, val, test = stratified_split(all_items, seed)
    for split_name, split_items in [("train", train), ("val", val), ("test", test)]:
        for src_path, cls_name in split_items:
            dst_dir = out_root / split_name / cls_name
            ensure_dir(dst_dir)
            dst_path = dst_dir / Path(src_path).name
            shutil.copy2(src_path, dst_path)

    # cleanup temporary
    shutil.rmtree(tmp_all)
    print(f"Saved {total_saved} cropped ROI images into {out_root}")


def _parse_args():
    p = argparse.ArgumentParser(description="Build labeled ROI dataset and split into train/val/test.")
    p.add_argument("--data-root", default="data", help="Path to dataset root containing group folders")
    p.add_argument("--out-root", default="dataset", help="Output dataset directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(Path(args.data_root), Path(args.out_root), seed=args.seed)


