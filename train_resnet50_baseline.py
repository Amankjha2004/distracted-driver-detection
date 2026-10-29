
from pathlib import Path
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(r"D:\AI_FINAL_PROJECT")
CSV_PATH = ROOT / "3mdad_master_metadata_split.csv"

OUTPUT_DIR = ROOT / "outputs" / "resnet50_pretrained_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STREAM = "Day-RGB1"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 20
NUM_CLASSES = 16
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

# --------------------------------------------------
# CHECK INPUT
# --------------------------------------------------

if not CSV_PATH.exists():
    raise SystemExit(f"Metadata CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required = {"image_path", "stream", "split", "activity_id"}
missing = required - set(df.columns)

if missing:
    raise SystemExit(f"Missing CSV columns: {sorted(missing)}")

# --------------------------------------------------
# SELECT BALANCED TRAINING AND VALIDATION SAMPLES
# --------------------------------------------------

def select_samples(split_name, count_per_class):
    subset = df[
        (df["stream"] == STREAM)
        & (df["split"] == split_name)
    ]

    selected = []

    for class_id in range(1, NUM_CLASSES + 1):
        class_rows = subset[subset["activity_id"] == class_id]

        if len(class_rows) < count_per_class:
            raise SystemExit(
                f"Not enough images for class {class_id} "
                f"in {split_name}."
            )

        selected.append(
            class_rows.sample(
                n=count_per_class,
                random_state=SEED
            )
        )

    result = pd.concat(selected)
    return result.sample(frac=1, random_state=SEED).reset_index(drop=True)

train_df = select_samples("train", TRAIN_PER_CLASS)
val_df = select_samples("validation", VAL_PER_CLASS)

print("=" * 65)
print("PRETRAINED RESNET50 BASELINE")
print("=" * 65)
print("Device:", DEVICE)
print("Stream:", STREAM)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Classes:", NUM_CLASSES)
print("Epochs:", EPOCHS)

# --------------------------------------------------
# IMAGE DATASET
# --------------------------------------------------

class DriverDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = Path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        # Convert IDs 1-16 to model indices 0-15.
        label = int(row["activity_id"]) - 1

        return image, label

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

train_dataset = DriverDataset(train_df, transform)
val_dataset = DriverDataset(val_df, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# LOAD PRETRAINED RESNET50
# --------------------------------------------------

print("\nLoading pretrained ResNet50...")

model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze all pretrained feature-extractor parameters.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace and train only the final classification layer.
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

print("Pretrained backbone frozen.")
print("Trainable parameters: classification layer only.")

# --------------------------------------------------
# TRAIN AND VALIDATE
# --------------------------------------------------

history = []
best_val_accuracy = -1.0
best_model_path = OUTPUT_DIR / "best_resnet50.pth"

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    elapsed = time.time() - epoch_start

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "epoch_seconds": elapsed,
    })

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train loss:       {train_loss:.4f}")
    print(f"Train accuracy:   {train_accuracy:.2%}")
    print(f"Validation loss:  {val_loss:.4f}")
    print(f"Validation acc.:  {val_accuracy:.2%}")
    print(f"Epoch duration:   {elapsed:.1f} seconds")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "stream": STREAM,
            "class_ids": list(range(1, NUM_CLASSES + 1)),
            "weights": "ImageNet pretrained ResNet50",
            "frozen_backbone": True,
            "seed": SEED,
        }, best_model_path)

        print("Best checkpoint saved.")

# --------------------------------------------------
# SAVE TRAINING HISTORY
# --------------------------------------------------

history_path = OUTPUT_DIR / "training_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

total_time = time.time() - start_time

print("\n" + "=" * 65)
print("BASELINE TRAINING COMPLETE")
print("=" * 65)
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print(f"Total duration: {total_time / 60:.1f} minutes")
print("Best model:", best_model_path)
print("Training history:", history_path)

print("\nThis is a balanced subset experiment.")
print("Do not report it as final full-dataset performance.")
