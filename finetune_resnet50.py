
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, f1_score, classification_report


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(r"D:\AI_FINAL_PROJECT")
CSV_PATH = ROOT / "3mdad_master_metadata_split.csv"

BASELINE_CHECKPOINT = (
    ROOT / "outputs" / "resnet50_pretrained_baseline"
    / "best_resnet50.pth"
)

OUTPUT_DIR = ROOT / "outputs" / "resnet50_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STREAM = "Day-RGB1"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3

TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 20
NUM_CLASSES = 16
SEED = 42

BACKBONE_LR = 0.0001
CLASSIFIER_LR = 0.001

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# CHECK INPUT FILES
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Metadata CSV not found: {CSV_PATH}")

if not BASELINE_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image_path", "stream", "split",
    "activity_id", "activity_label"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing CSV columns: {sorted(missing_columns)}"
    )


# --------------------------------------------------
# SELECT THE SAME BALANCED SUBSETS AS BASELINE
# --------------------------------------------------

def select_samples(split_name, count_per_class):
    subset = df[
        (df["stream"] == STREAM)
        & (df["split"] == split_name)
    ]

    selected = []

    for class_id in range(1, NUM_CLASSES + 1):
        class_rows = subset[
            subset["activity_id"] == class_id
        ]

        if len(class_rows) < count_per_class:
            raise ValueError(
                f"Not enough images for activity {class_id} "
                f"in split '{split_name}'."
            )

        selected.append(
            class_rows.sample(
                n=count_per_class,
                random_state=SEED
            )
        )

    result = pd.concat(selected)
    return result.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


train_df = select_samples("train", TRAIN_PER_CLASS)
val_df = select_samples("validation", VAL_PER_CLASS)


# Class IDs must map to model indices in the original order.
class_names = []

for class_id in range(1, NUM_CLASSES + 1):
    names = (
        df.loc[
            (df["stream"] == STREAM)
            & (df["activity_id"] == class_id),
            "activity_label"
        ]
        .dropna()
        .unique()
    )

    if len(names) != 1:
        raise ValueError(
            f"Activity ID {class_id} must map to exactly one label. "
            f"Found: {names.tolist()}"
        )

    class_names.append(str(names[0]))


# --------------------------------------------------
# IMAGE TRANSFORMS
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class DriverDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = Path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = transform(image)

        label = int(row["activity_id"]) - 1
        return image, label


train_loader = DataLoader(
    DriverDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    DriverDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# LOAD THE TRAINED BASELINE CHECKPOINT
# --------------------------------------------------

print("=" * 65)
print("RESNET50 FINE-TUNING")
print("=" * 65)
print("Device:", DEVICE)
print("Stream:", STREAM)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Classes:", NUM_CLASSES)
print("Epochs:", EPOCHS)

print("\nLoading baseline checkpoint...")

checkpoint = torch.load(
    BASELINE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

if checkpoint.get("num_classes") != NUM_CLASSES:
    raise ValueError("Checkpoint class count does not match.")

if checkpoint.get("stream") != STREAM:
    raise ValueError("Checkpoint stream does not match.")

if checkpoint.get("class_ids") != list(range(1, NUM_CLASSES + 1)):
    raise ValueError("Checkpoint class mapping does not match.")

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(checkpoint["model_state_dict"])

# Freeze all parameters first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final residual block and classification layer.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.layer4.parameters(),
        "lr": BACKBONE_LR
    },
    {
        "params": model.fc.parameters(),
        "lr": CLASSIFIER_LR
    }
])

print("Loaded trained baseline weights.")
print("Trainable layers: layer4 and fc")
print("Layer4 learning rate:", BACKBONE_LR)
print("Classifier learning rate:", CLASSIFIER_LR)


# --------------------------------------------------
# TRAINING AND VALIDATION HELPERS
# --------------------------------------------------

def run_training_epoch():
    model.train()

    # Keep earlier frozen batch-normalization layers in eval mode.
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    loss_sum = 0.0
    actual = []
    predicted = []

    for batch_index, (images, labels) in enumerate(
        train_loader, start=1
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        actual.extend(labels.cpu().numpy().tolist())
        predicted.extend(
            outputs.argmax(dim=1).detach().cpu().numpy().tolist()
        )

        if batch_index % 25 == 0 or batch_index == len(train_loader):
            print(
                f"  Train batches: {batch_index}/{len(train_loader)}",
                flush=True
            )

    return (
        loss_sum / len(actual),
        accuracy_score(actual, predicted),
        f1_score(actual, predicted, average="macro", zero_division=0)
    )


def run_validation():
    model.eval()

    loss_sum = 0.0
    actual = []
    predicted = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * labels.size(0)
            actual.extend(labels.cpu().numpy().tolist())
            predicted.extend(
                outputs.argmax(dim=1).cpu().numpy().tolist()
            )

    average_loss = loss_sum / len(actual)
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    return average_loss, accuracy, macro_f1, actual, predicted


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

best_val_f1 = -1.0
best_checkpoint_path = OUTPUT_DIR / "best_finetuned_resnet50.pth"
history = []

overall_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    print(f"\nEpoch {epoch + 1}/{EPOCHS}", flush=True)

    train_loss, train_acc, train_f1 = run_training_epoch()

    val_loss, val_acc, val_f1, _, _ = run_validation()

    epoch_seconds = time.time() - epoch_start

    print(f"Train loss:       {train_loss:.4f}")
    print(f"Train accuracy:   {train_acc:.2%}")
    print(f"Train macro F1:   {train_f1:.4f}")
    print(f"Validation loss:  {val_loss:.4f}")
    print(f"Validation acc.:  {val_acc:.2%}")
    print(f"Validation F1:    {val_f1:.4f}")
    print(f"Epoch duration:   {epoch_seconds:.1f} seconds")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
        "epoch_seconds": epoch_seconds
    })

    # Select checkpoint by validation macro F1.
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "stream": STREAM,
            "class_ids": list(range(1, NUM_CLASSES + 1)),
            "class_names": class_names,
            "seed": SEED,
            "backbone_learning_rate": BACKBONE_LR,
            "classifier_learning_rate": CLASSIFIER_LR,
            "epoch": epoch + 1,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1
        }, best_checkpoint_path)

        print("Best fine-tuned checkpoint saved.")

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

total_seconds = time.time() - overall_start

print("\n" + "=" * 65)
print("FINE-TUNING COMPLETE")
print("=" * 65)
print(f"Best validation macro F1: {best_val_f1:.4f}")
print(f"Total duration: {total_seconds / 60:.1f} minutes")
print("Best checkpoint:", best_checkpoint_path)
print("Training history:", OUTPUT_DIR / "training_history.csv")


# --------------------------------------------------
# FINAL REPORT FOR THE BEST CHECKPOINT
# --------------------------------------------------

print("\nEvaluating the best saved checkpoint...")

best_checkpoint = torch.load(
    best_checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(best_checkpoint["model_state_dict"])
model.eval()

_, final_acc, final_f1, y_true, y_pred = run_validation()

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(f"Best-checkpoint validation accuracy: {final_acc:.2%}")
print(f"Best-checkpoint validation macro F1: {final_f1:.4f}")
print("\nClassification report:\n")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Validation accuracy: {final_acc:.4%}\n")
    file.write(f"Validation macro F1: {final_f1:.4f}\n\n")
    file.write(report)

print("\nAll outputs saved to:", OUTPUT_DIR)
print("This remains a balanced-subset experiment.")
print("Do not report it as full-dataset test performance.")
