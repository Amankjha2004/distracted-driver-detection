
from pathlib import Path
import random
import time
import copy

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

START_CHECKPOINT = (
    ROOT / "outputs" / "resnet50_finetuned"
    / "best_finetuned_resnet50.pth"
)

OUTPUT_DIR = ROOT / "outputs" / "resnet50_finetuned_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STREAM = "Day-RGB1"
IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 5
PATIENCE = 2

TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 20
NUM_CLASSES = 16
SEED = 42

BACKBONE_LR = 0.00001
CLASSIFIER_LR = 0.0001
WEIGHT_DECAY = 0.0001

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# CHECK INPUTS
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Metadata CSV not found: {CSV_PATH}")

if not START_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Starting checkpoint not found: {START_CHECKPOINT}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image_path", "stream", "split",
    "activity_id", "activity_label"
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing metadata columns: {sorted(missing_columns)}"
    )


# --------------------------------------------------
# SELECT REPRODUCIBLE BALANCED SUBSETS
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
                f"in split {split_name}."
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


# Keep the output class order identical to training.
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
            f"Activity ID {class_id} must map to one label. "
            f"Found: {names.tolist()}"
        )

    class_names.append(str(names[0]))


# --------------------------------------------------
# IMAGE TRANSFORMS
# --------------------------------------------------

# Augmentation is applied only to training images.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
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
    def __init__(self, dataframe, image_transform):
        self.data = dataframe.reset_index(drop=True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        with Image.open(Path(row["image_path"])) as image:
            image = image.convert("RGB")
            image = self.image_transform(image)

        label = int(row["activity_id"]) - 1
        return image, label


train_loader = DataLoader(
    DriverDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    DriverDataset(val_df, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# LOAD PREVIOUS BEST MODEL
# --------------------------------------------------

print("=" * 65)
print("RESNET50 REGULARIZED FINE-TUNING V2")
print("=" * 65)
print("Device:", DEVICE)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Maximum epochs:", MAX_EPOCHS)
print("Early stopping patience:", PATIENCE)

checkpoint = torch.load(
    START_CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

if checkpoint.get("num_classes") != NUM_CLASSES:
    raise ValueError("Checkpoint class count does not match.")

if checkpoint.get("stream") != STREAM:
    raise ValueError("Checkpoint stream does not match.")

if checkpoint.get("class_ids") != list(range(1, NUM_CLASSES + 1)):
    raise ValueError("Checkpoint activity ID mapping does not match.")

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])

# Freeze everything first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Train only the final residual block and classifier.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.layer4.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.fc.parameters(),
            "lr": CLASSIFIER_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

print("\nLoaded starting checkpoint:", START_CHECKPOINT)
print("Trainable layers: layer4 and fc")
print("Backbone learning rate:", BACKBONE_LR)
print("Classifier learning rate:", CLASSIFIER_LR)
print("Weight decay:", WEIGHT_DECAY)


# --------------------------------------------------
# TRAINING AND VALIDATION
# --------------------------------------------------

def train_one_epoch():
    model.train()

    # Keep frozen earlier layers and their batch-normalization
    # statistics fixed during fine-tuning.
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    total_loss = 0.0
    actual = []
    predicted = []

    for batch_no, (images, labels) in enumerate(
        train_loader, start=1
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        actual.extend(labels.cpu().numpy().tolist())
        predicted.extend(
            outputs.argmax(dim=1).detach().cpu().numpy().tolist()
        )

        if batch_no % 25 == 0 or batch_no == len(train_loader):
            print(
                f"  Training batches: {batch_no}/{len(train_loader)}",
                flush=True
            )

    return (
        total_loss / len(actual),
        accuracy_score(actual, predicted),
        f1_score(
            actual, predicted,
            average="macro",
            zero_division=0
        )
    )


def validate():
    model.eval()

    total_loss = 0.0
    actual = []
    predicted = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            actual.extend(labels.cpu().numpy().tolist())
            predicted.extend(
                outputs.argmax(dim=1).cpu().numpy().tolist()
            )

    val_loss = total_loss / len(actual)
    val_accuracy = accuracy_score(actual, predicted)
    val_macro_f1 = f1_score(
        actual, predicted,
        average="macro",
        zero_division=0
    )

    return val_loss, val_accuracy, val_macro_f1, actual, predicted


# --------------------------------------------------
# TRAIN WITH EARLY STOPPING
# --------------------------------------------------

best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

best_checkpoint_path = OUTPUT_DIR / "best_finetuned_resnet50_v2.pth"
history = []

start_time = time.time()

for epoch in range(MAX_EPOCHS):
    epoch_start = time.time()

    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}", flush=True)

    train_loss, train_acc, train_f1 = train_one_epoch()

    val_loss, val_acc, val_f1, _, _ = validate()

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

    # Select best model by validation macro F1.
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "stream": STREAM,
            "class_ids": list(range(1, NUM_CLASSES + 1)),
            "class_names": class_names,
            "seed": SEED,
            "epoch": best_epoch,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
            "backbone_learning_rate": BACKBONE_LR,
            "classifier_learning_rate": CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY
        }, best_checkpoint_path)

        print("Best checkpoint saved.")
    else:
        epochs_without_improvement += 1
        print(
            "No macro F1 improvement:",
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# --------------------------------------------------
# REPORT BEST CHECKPOINT
# --------------------------------------------------

elapsed = time.time() - start_time

print("\n" + "=" * 65)
print("V2 TRAINING COMPLETE")
print("=" * 65)
print("Best epoch:", best_epoch)
print(f"Best validation macro F1: {best_val_f1:.4f}")
print(f"Total time: {elapsed / 60:.1f} minutes")
print("Checkpoint:", best_checkpoint_path)

# Reload the best checkpoint before final reporting.
best_checkpoint = torch.load(
    best_checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)
model.load_state_dict(best_checkpoint["model_state_dict"])

_, best_accuracy, best_f1, y_true, y_pred = validate()

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(f"\nBest-checkpoint validation accuracy: {best_accuracy:.2%}")
print(f"Best-checkpoint validation macro F1: {best_f1:.4f}")
print("\nClassification report:\n")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Validation accuracy: {best_accuracy:.4%}\n")
    file.write(f"Validation macro F1: {best_f1:.4f}\n\n")
    file.write(report)

print("\nResults saved to:", OUTPUT_DIR)
print("This is a balanced-subset experiment, not final test performance.")
