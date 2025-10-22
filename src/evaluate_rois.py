"""
GOAL: Evaluate ROI detection performance and quality metrics
- Analyze detected ROIs vs ground truth annotations
- Calculate precision, recall, F1-score for defect detection
- Generate detection quality reports and visualizations
- Compare different preprocessing parameters
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import csv

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.ops import nms


# ---------- Paths and constants ----------

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------- Utilities for dataset layout ----------

def infer_label_path_from_test(test_path: Path) -> Path:
    parent = test_path.parent
    stem = test_path.stem
    base_id = stem.split("_")[0]
    label_dir = parent.parent / f"{parent.name}_not"
    return label_dir / f"{base_id}.txt"


def read_gt_labels(label_path: Path) -> List[Tuple[int, int, int, int, int]]:
    labels: List[Tuple[int, int, int, int, int]] = []
    if not label_path.exists():
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2, cls_id = map(int, parts[:5])
            labels.append((x1, y1, x2, y2, cls_id))
    return labels


def list_test_pairs(data_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for group_dir in sorted(data_root.glob("group*/")):
        for board_dir in sorted(group_dir.glob("*/")):
            # look for *_temp.jpg and *_test.jpg pairs
            temp_imgs = sorted(board_dir.glob("*_temp.jpg"))
            for temp_img in temp_imgs:
                base_id = temp_img.stem.split("_")[0]
                test_img = board_dir / f"{base_id}_test.jpg"
                if test_img.exists():
                    pairs.append((temp_img, test_img))
    return pairs


# ---------- Preprocessing and ROI extraction ----------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    g = cv2.equalizeHist(g)
    return g


def absdiff_norm(template_gray: np.ndarray, test_gray: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(test_gray, template_gray)
    return diff


def mask_from_diff(diff: np.ndarray, thresh: int = 30) -> np.ndarray:
    _, th = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def extract_rois(mask: np.ndarray, min_area: int = 50) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= min_area:
            rois.append((x, y, w, h))
    rois.sort(key=lambda b: b[2] * b[3], reverse=True)
    return rois


# ---------- Model ----------

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[-1].in_features  # type: ignore[attr-defined]
    model.classifier[-1] = nn.Linear(in_feats, num_classes)  # type: ignore[index]
    return model


def load_classes(classes_file: Path) -> List[str]:
    with open(classes_file, "r", encoding="utf-8") as f:
        classes: List[str] = json.load(f)
    return classes


def transform_for_model(img_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(MEAN, STD),
    ])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return tfm(img_rgb)  # type: ignore[arg-type]
 

# ---------- Matching and metrics ----------

def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return (x, y, x + w, y + h)


def compute_metrics(matches: List[Dict[str, Any]], classes: List[str]) -> Dict[str, Any]:
    # Per-class TP, FP, FN
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    tp = np.zeros(len(classes), dtype=np.int64)
    fp = np.zeros(len(classes), dtype=np.int64)
    fn = np.zeros(len(classes), dtype=np.int64)

    num_pred = 0
    num_gt = 0
    num_correct = 0

    for m in matches:
        gt_label = m.get("gt_label")
        pred_label = m.get("pred_label")
        matched = m.get("matched", False)
        if gt_label is not None:
            num_gt += 1
        if pred_label is not None:
            num_pred += 1
        if matched and gt_label is not None and pred_label is not None and gt_label == pred_label:
            num_correct += 1

        if gt_label is None and pred_label is not None:
            fp[cls_to_idx[pred_label]] += 1
        elif gt_label is not None and pred_label is None:
            fn[cls_to_idx[gt_label]] += 1
        elif gt_label is not None and pred_label is not None:
            if gt_label == pred_label:
                tp[cls_to_idx[gt_label]] += 1
            else:
                fp[cls_to_idx[pred_label]] += 1
                fn[cls_to_idx[gt_label]] += 1

    precision = (tp / np.maximum(tp + fp, 1)).tolist()
    recall = (tp / np.maximum(tp + fn, 1)).tolist()
    f1 = ((2 * tp) / np.maximum(2 * tp + fp + fn, 1)).tolist()

    overall = {
        "prediction_match_rate": (num_correct / max(num_gt, 1)),
        "false_positive_rate": (fp.sum() / max(num_pred, 1)),
        "false_negative_rate": (fn.sum() / max(num_gt, 1)),
    }

    per_class = {
        c: {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
        }
        for i, c in enumerate(classes)
    }

    return {
        "overall": overall,
        "per_class": per_class,
        "totals": {"num_pred": int(num_pred), "num_gt": int(num_gt), "num_correct": int(num_correct)},
    }


# ---------- Core evaluation ----------

def evaluate(
    data_root: Path,
    classes_file: Path,
    model_path: Path,
    out_dir: Path,
    img_size: int = 128,
    device_str: str | None = None,
    mask_thresh: int = 30,
    min_area: int = 50,
    iou_thresh: float = 0.3,
    conf_thresh: float = 0.6,
    nms_iou_thresh: float = 0.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(classes_file)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pairs = list_test_pairs(data_root)
    print(f"Found {len(pairs)} template/test pairs under {data_root}")

    detailed_rows: List[Dict[str, Any]] = []
    matches_all: List[Dict[str, Any]] = []

    with torch.no_grad():
        for temp_path, test_path in pairs:
            test_img = cv2.imread(str(test_path))
            temp_img = cv2.imread(str(temp_path))
            if test_img is None or temp_img is None:
                continue

            gt_labels = read_gt_labels(infer_label_path_from_test(test_path))

            gt_boxes_xyxy = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _c) in gt_labels]
            gt_classes = [_c for (_x1, _y1, _x2, _y2, _c) in gt_labels]

            g_temp = preprocess_gray(to_gray(temp_img))
            g_test = preprocess_gray(to_gray(test_img))
            diff = absdiff_norm(g_temp, g_test)
            mask = mask_from_diff(diff, thresh=mask_thresh)
            rois = extract_rois(mask, min_area=min_area)

            pred_boxes_xyxy: List[Tuple[int, int, int, int]] = []
            pred_classes: List[str] = []
            pred_scores: List[float] = []

            for (x, y, w, h) in rois:
                crop = test_img[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                t = transform_for_model(crop, img_size).unsqueeze(0).to(device)
                logits = model(t)
                probs = torch.softmax(logits, dim=1)[0]
                score, pred_idx = torch.max(probs, 0)
                pred_label = classes[int(pred_idx.item())]
                pred_boxes_xyxy.append(xywh_to_xyxy(x, y, w, h))
                pred_classes.append(pred_label)
                pred_scores.append(float(score.item()))

            # Apply confidence filter and class-wise NMS
            keep_indices: List[int] = []
            for cls_name in classes:
                idxs = [i for i, c in enumerate(pred_classes) if c == cls_name and pred_scores[i] >= conf_thresh]
                if not idxs:
                    continue
                boxes_tensor = torch.tensor([pred_boxes_xyxy[i] for i in idxs], dtype=torch.float32)
                scores_tensor = torch.tensor([pred_scores[i] for i in idxs], dtype=torch.float32)
                keep_local = nms(boxes_tensor, scores_tensor, nms_iou_thresh)
                keep_indices.extend([idxs[int(k)] for k in keep_local])

            # Sort kept indices by score descending to make visualization consistent
            keep_indices = sorted(keep_indices, key=lambda i: pred_scores[i], reverse=True)
            pred_boxes_xyxy = [pred_boxes_xyxy[i] for i in keep_indices]
            pred_classes = [pred_classes[i] for i in keep_indices]
            pred_scores = [pred_scores[i] for i in keep_indices]

            # Matching predictions to GT by IoU greedy
            matched_gt = [False] * len(gt_boxes_xyxy)
            matched_pred = [False] * len(pred_boxes_xyxy)

            for pi, p_box in enumerate(pred_boxes_xyxy):
                best_iou = 0.0
                best_gi = -1
                for gi, g_box in enumerate(gt_boxes_xyxy):
                    if matched_gt[gi]:
                        continue
                    i = iou(p_box, g_box)
                    if i > best_iou:
                        best_iou = i
                        best_gi = gi
                if best_gi >= 0 and best_iou >= iou_thresh:
                    matched_pred[pi] = True
                    matched_gt[best_gi] = True
                    gt_label_name = class_id_to_name(gt_classes[best_gi], classes)
                    matches_all.append({
                        "image": str(test_path),
                        "pred_box": p_box,
                        "pred_label": pred_classes[pi],
                        "gt_box": gt_boxes_xyxy[best_gi],
                        "gt_label": gt_label_name,
                        "matched": pred_classes[pi] == gt_label_name,
                        "iou": best_iou,
                        "score": pred_scores[pi],
                    })
                else:
                    matches_all.append({
                        "image": str(test_path),
                        "pred_box": p_box,
                        "pred_label": pred_classes[pi],
                        "gt_box": None,
                        "gt_label": None,
                        "matched": False,
                        "iou": 0.0,
                        "score": pred_scores[pi],
                    })

            # Unmatched GT -> FN entries
            for gi, g_box in enumerate(gt_boxes_xyxy):
                if not matched_gt[gi]:
                    matches_all.append({
                        "image": str(test_path),
                        "pred_box": None,
                        "pred_label": None,
                        "gt_box": g_box,
                        "gt_label": class_id_to_name(gt_classes[gi], classes),
                        "matched": False,
                        "iou": 0.0,
                        "score": None,
                    })

            # Draw annotated output
            annotated = test_img.copy()
            # draw GT in green
            for (x1, y1, x2, y2), cls_id in zip(gt_boxes_xyxy, gt_classes):
                name = class_id_to_name(cls_id, classes)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, name, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # draw predictions in red
            for (p_box, name, score) in zip(pred_boxes_xyxy, pred_classes, pred_scores):
                x1, y1, x2, y2 = p_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{name} {score:.2f}", (x1, min(annotated.shape[0]-2, y2+16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            rel = test_path.relative_to(data_root)
            out_img_dir = out_dir / "annotated" / rel.parent
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = out_img_dir / f"{test_path.stem}_annotated.jpg"
            cv2.imwrite(str(out_img_path), annotated)

            # Detailed rows for CSV
            for (p_box, name, score) in zip(pred_boxes_xyxy, pred_classes, pred_scores):
                detailed_rows.append({
                    "image": str(test_path),
                    "pred_x1": p_box[0], "pred_y1": p_box[1], "pred_x2": p_box[2], "pred_y2": p_box[3],
                    "pred_label": name, "score": score,
                })
            for (g_box, cls_id) in zip(gt_boxes_xyxy, gt_classes):
                detailed_rows.append({
                    "image": str(test_path),
                    "gt_x1": g_box[0], "gt_y1": g_box[1], "gt_x2": g_box[2], "gt_y2": g_box[3],
                    "gt_label": class_id_to_name(cls_id, classes),
                })

    # Metrics
    metrics = compute_metrics(matches_all, classes)

    # Save metrics JSON
    with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save matches JSON for debugging
    with open(out_dir / "detailed_matches.json", "w", encoding="utf-8") as f:
        json.dump(matches_all, f, indent=2)

    # Save CSV
    csv_path = out_dir / "predictions_and_gt.csv"
    fieldnames = sorted({k for row in detailed_rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)

    # Console summary
    overall = metrics["overall"]
    print("\n=== Evaluation Summary ===")
    print(f"Pairs evaluated: {len(pairs)}")
    print(f"Prediction match rate: {overall['prediction_match_rate']:.4f}")
    print(f"False positive rate:   {overall['false_positive_rate']:.4f}")
    print(f"False negative rate:   {overall['false_negative_rate']:.4f}")
    print(f"Saved annotated images to: {out_dir / 'annotated'}")
    print(f"Saved metrics to: {out_dir / 'evaluation_metrics.json'} and {csv_path}")


def class_id_to_name(cls_id: int, classes: List[str]) -> str:
    # Mapping in data labels uses 1..6; training classes.json holds names in some order
    # Attempt direct mapping by class name equality when possible; else map by an index list
    # Known dataset order used in classes.json
    # ["mousebite","open","pinhole","short","spur","spurious copper"]
    id_to_name = {
        1: "open",
        2: "short",
        3: "mousebite",
        4: "spur",
        5: "pinhole",
        6: "spurious copper",
    }
    name = id_to_name.get(cls_id, f"class_{cls_id}")
    if name in classes:
        return name
    return name


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate EfficientNet-B4 on extracted ROIs from test images")
    p.add_argument("--data", default="data", help="Root folder containing group*/ subfolders")
    p.add_argument("--classes", default="training_outputs/classes.json", help="Path to classes.json")
    p.add_argument("--model", default="training_outputs/model_best.pth", help="Path to trained model weights")
    p.add_argument("--out", default="evaluation_outputs", help="Output directory")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--thresh", type=int, default=30, help="Threshold for diff -> mask")
    p.add_argument("--min-area", type=int, default=50, help="Minimum ROI area in pixels")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching predictions to GT")
    p.add_argument("--conf", type=float, default=0.6, help="Confidence threshold for showing detections")
    p.add_argument("--nms-iou", type=float, default=0.5, help="IoU threshold for NMS")
    return p.parse_args()


def main():
    args = _parse_args()
    evaluate(
        data_root=Path(args.data),
        classes_file=Path(args.classes),
        model_path=Path(args.model),
        out_dir=Path(args.out),
        img_size=args.img_size,
        device_str=args.device,
        mask_thresh=args.thresh,
        min_area=args.min_area,
        iou_thresh=args.iou,
        conf_thresh=args.conf,
        nms_iou_thresh=args.nms_iou,
    )


if __name__ == "__main__":
    main()


