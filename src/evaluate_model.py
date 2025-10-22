"""
GOAL: Evaluate trained EfficientNet-B4 model on test set
- Load trained model weights and test on unseen data
- Calculate test accuracy and per-class performance
- Generate confusion matrix visualization
- Save detailed results and classification report
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# TEST DATA LOADING: Create test dataloader for evaluation
def create_test_dataloader(data_dir: Path, img_size: int, batch_size: int) -> tuple[DataLoader, list[str]]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_ds = datasets.ImageFolder(str(data_dir / 'test'), transform=test_tf)
    class_names = test_ds.classes

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader, class_names


# MODEL RECONSTRUCTION: Build same model architecture for loading weights
def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes) # type: ignore
    return model


# MODEL EVALUATION: Test model and generate confusion matrix
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, class_names: list[str], out_dir: Path):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())
    
    # Calculate test accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    test_accuracy = correct / len(y_true)
    print(f"Test Accuracy: {test_accuracy:.4f} ({correct}/{len(y_true)})")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, colorbar=True, xticks_rotation=45)
    plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.4f}')
    plt.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'test_confusion_matrix.jpg', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save detailed results
    results = {
        'test_accuracy': test_accuracy,
        'total_samples': len(y_true),
        'correct_predictions': correct,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'classification_report': report
    }
    
    with open(out_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print per-class accuracy
    print("\nPer-class Test Accuracy:")
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"{class_name}: {class_acc:.4f} ({class_correct}/{class_total})")
    
    print(f"\nResults saved to {out_dir}")
    return test_accuracy, cm


def _parse_args():
    p = argparse.ArgumentParser(description='Evaluate trained EfficientNet-B4 on test set')
    p.add_argument('--data', default='dataset', help='Path to dataset directory')
    p.add_argument('--model', default='training_outputs/model_best.pth', help='Path to trained model')
    p.add_argument('--out', default='evaluation_outputs', help='Output directory for results')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = _parse_args()
    data_dir = Path(args.data)
    model_path = Path(args.model)
    out_dir = Path(args.out)
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return
    
    # Load class names
    classes_file = model_path.parent / 'classes.json'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = json.load(f)
    else:
        # Fallback: get from dataset
        test_loader, class_names = create_test_dataloader(data_dir, args.img_size, args.batch_size)
    
    test_loader, class_names = create_test_dataloader(data_dir, args.img_size, args.batch_size)
    
    device = torch.device(args.device)
    model = build_model(num_classes=len(class_names)).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"Evaluating model on test set...")
    print(f"Device: {device}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    test_accuracy, confusion_mat = evaluate_model(model, test_loader, device, class_names, out_dir)
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    main()
