"""
MobileNetV3 + Attention experiment for 3MDAD Day-RGB1.
CPU-friendly transfer-learning setup using the existing V2 train/validation folders.
The held-out test set is never loaded or used.

Run from PowerShell:
    cd D:\AI_FINAL_PROJECT
    .\.venv\Scripts\python.exe train_mobilenetv3_attention.py
"""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------- Configuration -----------------------------
ROOT = Path(r"D:\AI_FINAL_PROJECT")
DATA_ROOT = ROOT / "outputs" / "yolov12_cnn_v2" / "dataset_v2"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
OUT_DIR = ROOT / "outputs" / "mobilenetv3_attention"

IMAGE_SIZE = 192
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42
NUM_WORKERS = 0  # safest setting for Windows
DEVICE = torch.device("cpu")

# -------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

print("=" * 68)
print("MobileNetV3 + Attention | 3MDAD Day-RGB1 | CPU")
print("Using prepared V2 dataset folders; held-out test data stays untouched.")
print(f"Train folder: {TRAIN_DIR}")
print(f"Validation folder: {VAL_DIR}")
print("=" * 68)

if not TRAIN_DIR.is_dir() or not VAL_DIR.is_dir():
    raise FileNotFoundError(
        "Prepared dataset folders were not found. Expected:\n"
        f"{TRAIN_DIR}\n{VAL_DIR}"
    )

# MobileNetV3 pretrained input normalization.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.0),  # preserve left/right driving labels
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_transform)

if train_ds.classes != val_ds.classes:
    raise ValueError("Train and validation class folders do not match.")

class_names = train_ds.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")
print(f"Train images: {len(train_ds)} | Validation images: {len(val_ds)}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)


class ChannelAttention(nn.Module):
    """Lightweight squeeze-and-excitation style channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.mlp(self.pool(x))


class SpatialAttention(nn.Module):
    """Learns which spatial locations in the feature map are important."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True).values
        attention = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention


class MobileNetV3Attention(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        base = models.mobilenet_v3_large(weights=weights)
        self.features = base.features
        channels = base.classifier[0].in_features  # 960 for MobileNetV3-Large

        # Freeze the backbone to make this experiment practical on CPU.
        for parameter in self.features.parameters():
            parameter.requires_grad = False

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.30),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.pool(x)
        return self.classifier(x)


model = MobileNetV3Attention(num_classes).to(DEVICE)

# Class weights help if the supplied folder counts are not perfectly balanced.
train_targets = np.array([label for _, label in train_ds.samples])
classes_index = np.unique(train_targets)
weights_np = compute_class_weight(
    class_weight="balanced", classes=classes_index, y=train_targets
)
class_weights = torch.ones(num_classes, dtype=torch.float32)
for index, weight in zip(classes_index, weights_np):
    class_weights[int(index)] = float(weight)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

best_f1 = -1.0
best_epoch = 0
best_path = OUT_DIR / "best_mobilenetv3_attention.pth"
history = []
start_all = time.time()


def evaluate():
    model.eval()
    all_true, all_pred = [], []
    total_loss, total_items = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_items += labels.size(0)
            predictions = outputs.argmax(dim=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(predictions.cpu().numpy().tolist())
    return (
        total_loss / max(1, total_items),
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro", zero_division=0),
        f1_score(all_true, all_pred, average="weighted", zero_division=0),
        all_true,
        all_pred,
    )


for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    # Keep the pretrained backbone in eval mode; only attention/head are trained.
    model.features.eval()
    running_loss, seen, correct = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

        if batch_idx % 50 == 0 or batch_idx == len(train_loader):
            print(
                f"Epoch {epoch}/{EPOCHS} | train batch "
                f"{batch_idx}/{len(train_loader)}"
            )

    train_loss = running_loss / max(1, seen)
    train_acc = correct / max(1, seen)
    val_loss, val_acc, val_macro_f1, val_weighted_f1, y_true, y_pred = evaluate()
    scheduler.step(val_macro_f1)

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_macro_f1,
        "validation_weighted_f1": val_weighted_f1,
        "epoch_minutes": (time.time() - epoch_start) / 60,
    }
    history.append(record)
    print(
        f"Epoch {epoch}: train_acc={train_acc:.4f}, "
        f"val_acc={val_acc:.4f}, macro-F1={val_macro_f1:.4f}, "
        f"weighted-F1={val_weighted_f1:.4f}, "
        f"time={record['epoch_minutes']:.2f} min"
    )

    # Select by macro-F1, which gives each class equal importance.
    if val_macro_f1 > best_f1:
        best_f1 = val_macro_f1
        best_epoch = epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "best_epoch": best_epoch,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_macro_f1,
            "validation_weighted_f1": val_weighted_f1,
        }, best_path)

# Restore the best checkpoint and generate final artifacts.
checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
_, final_acc, final_macro_f1, final_weighted_f1, y_true, y_pred = evaluate()

report_text = classification_report(
    y_true, y_pred, labels=list(range(num_classes)),
    target_names=class_names, digits=4, zero_division=0
)
(OUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
with (OUT_DIR / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true/predicted"] + class_names)
    for name, row in zip(class_names, cm):
        writer.writerow([name] + row.tolist())

with (OUT_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

elapsed_minutes = (time.time() - start_all) / 60
results = {
    "experiment": "MobileNetV3 + Attention",
    "dataset": "3MDAD Day-RGB1",
    "backbone": "ImageNet-pretrained MobileNetV3-Large (frozen)",
    "attention": "Channel attention + spatial attention",
    "train_images": len(train_ds),
    "validation_images": len(val_ds),
    "classes": num_classes,
    "image_size": IMAGE_SIZE,
    "epochs_run": EPOCHS,
    "best_epoch_by_macro_f1": best_epoch,
    "validation_accuracy": float(final_acc),
    "validation_macro_f1": float(final_macro_f1),
    "validation_weighted_f1": float(final_weighted_f1),
    "elapsed_minutes": round(elapsed_minutes, 2),
    "model_path": str(best_path),
    "held_out_test_used": False,
}
(OUT_DIR / "final_results.txt").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
comparison_path = OUT_DIR / "model_comparison.csv"
with comparison_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "validation_accuracy", "macro_f1", "weighted_f1"])
    writer.writerow([
        results["experiment"], f"{final_acc:.6f}",
        f"{final_macro_f1:.6f}", f"{final_weighted_f1:.6f}"
    ])

print("\nMOBILENETV3 + ATTENTION COMPLETED")
print(json.dumps(results, indent=2))
print(f"Results: {OUT_DIR / 'final_results.txt'}")
print(f"Classification report: {OUT_DIR / 'classification_report.txt'}")
print(f"Confusion matrix: {OUT_DIR / 'confusion_matrix.csv'}")
print(f"Training history: {OUT_DIR / 'training_history.csv'}")
