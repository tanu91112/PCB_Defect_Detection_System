"""
GOAL: Train EfficientNet-B4 classifier on PCB defect ROI dataset
- Load pre-trained EfficientNet-B4 and fine-tune for 6 defect classes
- Apply data augmentation (flip, rotation, normalization)
- Train with Adam optimizer and CrossEntropyLoss
- Track training/validation metrics and save best model
- Generate accuracy/loss curves and confusion matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import copy
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# DATA LOADING: Create train/val/test dataloaders with augmentation
def create_dataloaders(data_dir: Path, img_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(str(data_dir / 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_dir / 'val'), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(data_dir / 'test'), transform=eval_tf)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, class_names

from torch import nn
from torchvision import models
from typing import cast

from torch import nn
from torchvision import models

# MODEL BUILDING: Create EfficientNet-B4 with custom classifier
def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    # Replace last Linear layer
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)  # type: ignore
    return model


# TRAINING LOOP: Train model with validation and early stopping
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, lr: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        n_samples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            n_samples += inputs.size(0)

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects / n_samples
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_samples += inputs.size(0)

        v_loss = val_loss / val_samples
        v_acc = val_corrects / val_samples
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:02d}: train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


# VISUALIZATION: Plot training/validation curves
def plot_curves(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curve.jpg', dpi=200)
    plt.close()
    # Acc
    plt.figure(figsize=(6,4))
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'accuracy_curve.jpg', dpi=200)
    plt.close()


# CONFUSION MATRIX: Generate and save confusion matrix visualization
def evaluate_confusion(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str], out_dir: Path):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(cmap='Blues', ax=ax, colorbar=False, xticks_rotation=45)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'confusion_matrix.jpg', dpi=200)
    plt.close()


def _parse_args():
    p = argparse.ArgumentParser(description='Train EfficientNet-B4 on ROI dataset')
    p.add_argument('--data', default='dataset', help='Path to dataset/{train,val,test}')
    p.add_argument('--out', default='training_outputs', help='Output directory')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = _parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(data_dir, args.img_size, args.batch_size)
    with open(out_dir / 'classes.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2)

    device = torch.device(args.device)
    model = build_model(num_classes=len(class_names)).to(device)
    model, history = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)

    torch.save(model.state_dict(), out_dir / 'model_best.pth')
    plot_curves(history, out_dir)
    evaluate_confusion(model, test_loader, device, class_names, out_dir)
    print(f"Saved model and plots to {out_dir}")


if __name__ == '__main__':
    main()


