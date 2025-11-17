"""Image preprocessing and defect ROI extraction for PCB defect dataset.

This script provides functions to:
- load images
- basic preprocessing (grayscale, blur, optional histogram equalization)
- compute absolute difference between template and test
- create a binary mask using thresholding + morphology
- extract contours and ROI bounding boxes, filter by area
- save cropped ROI images and annotated preview

Usage (from workspace root or anywhere):
python src/preprocessing.py --template "data/group00041/00041/00041000_temp.jpg" --test "data/group00041/00041/00041000_test.jpg" --out "EDA_Output/preprocess_example"

The script writes: diff.jpg, mask.jpg, annotated.jpg and cropped ROI images into the output folder.

Requires: opencv-python, numpy
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
from typing import List, Tuple

# Mapping for the 6-class PCB defect dataset
DEFECT_CLASSES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "pinhole",
    6: "spurious copper",
}

# Infer label path based on test image
def _infer_label_path_from_test(test_path: str) -> Path:
    p = Path(test_path)
    parent = p.parent
    stem = p.stem
    base_id = stem.split("_")[0]
    label_dir = parent.parent / f"{parent.name}_not"
    return label_dir / f"{base_id}.txt"

# Read labels from label file
def read_labels(label_path: Path) -> List[Tuple[int, int, int, int, int]]:
    labels: List[Tuple[int, int, int, int, int]] = []
    if not label_path.exists():
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2, cls = map(int, parts[:5])
            labels.append((x1, y1, x2, y2, cls))
    return labels

# Load an image
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

# Convert to grayscale
def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocess grayscale image
def preprocess_gray(gray: np.ndarray, blur_ksize: Tuple[int, int] = (5, 5), equalize: bool = True) -> np.ndarray:
    g = cv2.GaussianBlur(gray, blur_ksize, 0)
    if equalize:
        g = cv2.equalizeHist(g)
    return g

def align_affine_ecc(template_gray: np.ndarray, test_gray: np.ndarray, test_color: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = template_gray.shape
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        cv2.findTransformECC(template_gray, test_gray, warp, cv2.MOTION_AFFINE, criteria)
        aligned_gray = cv2.warpAffine(test_gray, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        aligned_color = cv2.warpAffine(test_color, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return aligned_gray, aligned_color
    except Exception:
        return test_gray, test_color

# Absolute difference between template and test
def subtract_images(template: np.ndarray, test: np.ndarray) -> np.ndarray:
    return cv2.absdiff(test, template)

# Create binary mask
def get_binary_mask(diff: np.ndarray, thresh: int = 30, morph_kernel: Tuple[int, int] = (5, 5)) -> np.ndarray:
    _, th = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated

# Extract ROI bounding boxes
def extract_rois(mask: np.ndarray, min_area: int = 50) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= min_area:
            rois.append((x, y, w, h))
    rois.sort(key=lambda b: b[2] * b[3], reverse=True)
    return rois

# Save cropped ROI images
def save_rois(image: np.ndarray, rois: List[Tuple[int, int, int, int]], out_dir: Path, prefix: str = "roi") -> List[Path]:
    out_paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y, w, h) in enumerate(rois):
        crop = image[y:y+h, x:x+w]
        p = out_dir / f"{prefix}_{i:03d}.jpg"
        cv2.imwrite(str(p), crop)
        out_paths.append(p)
    return out_paths

# Annotate image with class labels if available
def annotate_image(image: np.ndarray, rois: List[Tuple[int, int, int, int]], labels: List[Tuple[int, int, int, int, int]] = None) -> np.ndarray: # type: ignore
    annotated = image.copy()
    if labels:
        for (x1, y1, x2, y2, cls_id) in labels:
            label_name = DEFECT_CLASSES.get(cls_id, f"class_{cls_id}")
            color = (0, 255, 0) if cls_id != 2 else (0, 0, 255)  # red for "short"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label_name, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        for i, (x, y, w, h) in enumerate(rois):
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(annotated, f"{i}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated

# Run preprocessing pipeline
def run_pipeline(template_path: str, test_path: str, out_dir: str, *, thresh: int = 30, min_area: int = 50, align: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading template: {template_path}")
    t = load_image(template_path)
    print(f"Loading test:     {test_path}")
    s = load_image(test_path)

    # Grayscale + preprocess
    gt = to_gray(t)
    gs = to_gray(s)
    p_t = preprocess_gray(gt)
    p_s = preprocess_gray(gs)
    if align:
        p_s, s = align_affine_ecc(p_t, p_s, s)

    # Difference + save diff image
    diff = subtract_images(p_t, p_s)
    diff_vis = np.zeros_like(diff)
    cv2.normalize(diff, diff_vis, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(out / "diff.jpg"), diff_vis)
    print(f"Saved diff image to {out / 'diff.jpg'}")

    # Binary mask
    mask = get_binary_mask(diff, thresh=thresh)
    cv2.imwrite(str(out / "mask.jpg"), mask)
    print(f"Saved mask to {out / 'mask.jpg'}")

    # ROIs + save crops
    rois = extract_rois(mask, min_area=min_area)
    print(f"Found {len(rois)} ROIs (area >= {min_area})")
    rois_dir = out / "rois"
    saved = save_rois(s, rois, rois_dir, prefix="roi")
    if saved:
        print(f"Saved {len(saved)} ROI crops to {rois_dir}")

    # Load labels if available
    label_path = _infer_label_path_from_test(test_path)
    labels = read_labels(label_path)
    print(f"Loaded {len(labels)} labels from {label_path}" if labels else "No label file found.")

    # Annotated image with class names
    annotated = annotate_image(s, rois, labels)
    cv2.imwrite(str(out / "annotated.jpg"), annotated)
    print(f"Saved annotated preview to {out / 'annotated.jpg'}")

    return {
        'diff': out / 'diff.jpg',
        'mask': out / 'mask.jpg',
        'annotated': out / 'annotated.jpg',
        'rois': saved,
    }

# Parse arguments
def _parse_args():
    p = argparse.ArgumentParser(description="Preprocess two PCB images (template/test), subtract, extract contours, and save ROIs.")
    p.add_argument('--template', '-t', required=True, help='Template image path (defect-free)')
    p.add_argument('--test', '-s', required=True, help='Test image path (possibly defective)')
    p.add_argument('--out', '-o', default='EDA_Output/preprocessing', help='Output directory')
    p.add_argument('--thresh', type=int, default=30, help='Threshold for diff -> mask')
    p.add_argument('--min-area', type=int, default=50, help='Minimum ROI area in pixels')
    p.add_argument('--align', action='store_true', help='Align test to template before subtraction')
    return p.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    run_pipeline(args.template, args.test, args.out, thresh=args.thresh, min_area=args.min_area, align=args.align)
