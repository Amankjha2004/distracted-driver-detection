
import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\AI_FINAL_PROJECT"

CSV_PATH = os.path.join(
    PROJECT_DIR, "3mdad_master_metadata_split.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "outputs", "efficientnetb3_vit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

STREAM_NAME = "Day-RGB1"

# Match the previous ResNet50 experiment.
TRAIN_IMAGES_PER_CLASS = 100
VAL_IMAGES_PER_CLASS = 20

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42

NUM_CLASSES = 16

ACTIVITIES = {
    1: "safe_driving",
    2: "hair_makeup",
    3: "adjusting_radio",
    4: "gps_operating",
    5: "texting_right_hand",
    6: "texting_left_hand",
    7: "phone_right_hand",
    8: "phone_left_hand",
    9: "taking_picture",
    10: "talking_to_passenger",
    11: "singing_or_dancing",
    12: "fatigue_somnolence",
    13: "drinking_right_hand",
    14: "drinking_left_hand",
    15: "reaching_behind",
    16: "smoking",
}

CLASS_NAMES = [
    ACTIVITIES[i] for i in range(1, NUM_CLASSES + 1)
]

# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 65)
print("EFFICIENTNETB3 + VISION TRANSFORMER")
print("=" * 65)
print("Device:", DEVICE)
print("Metadata:", CSV_PATH)
print("Output:", OUTPUT_DIR)

# ============================================================
# 3. LOAD AND PREPARE METADATA
# ============================================================

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"Metadata CSV not found:\n{CSV_PATH}\n"
        "Check the filename and project folder."
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image_path",
    "stream",
    "split",
    "activity_id",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required CSV columns: {missing_columns}"
    )

df["stream"] = df["stream"].astype(str).str.strip()
df["split"] = (
    df["split"].astype(str).str.strip().str.lower()
)
df["activity_id"] = pd.to_numeric(
    df["activity_id"], errors="coerce"
)

df = df.dropna(
    subset=["image_path", "stream", "split", "activity_id"]
).copy()

df["activity_id"] = df["activity_id"].astype(int)

# Select only the Day-RGB1 stream.
df = df[df["stream"] == STREAM_NAME].copy()

if df.empty:
    raise ValueError(
        f"No rows found for stream '{STREAM_NAME}'."
    )

# Standardize common split-name variations.
split_aliases = {
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "testing": "test",
}

df["split"] = df["split"].replace(split_aliases)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

if train_df.empty or val_df.empty:
    raise ValueError(
        "Training or validation rows are missing. "
        "Check the values in the CSV 'split' column."
    )

# The classifier uses labels 0-15, corresponding to activity IDs 1-16.
train_df["label"] = train_df["activity_id"] - 1
val_df["label"] = val_df["activity_id"] - 1

valid_labels = set(range(NUM_CLASSES))

if not set(train_df["label"].unique()).issubset(valid_labels):
    raise ValueError("Unexpected activity IDs in training data.")

if not set(val_df["label"].unique()).issubset(valid_labels):
    raise ValueError("Unexpected activity IDs in validation data.")

# Sample the same number of images per class.
def balanced_sample(dataframe, samples_per_class, split_name):
    selected = []

    for label in range(NUM_CLASSES):
        class_df = dataframe[dataframe["label"] == label]

        if len(class_df) < samples_per_class:
            raise ValueError(
                f"{split_name}: class {label + 1} "
                f"has only {len(class_df)} images; "
                f"{samples_per_class} are required."
            )

        sampled = class_df.sample(
            n=samples_per_class,
            random_state=RANDOM_SEED,
        )

        selected.append(sampled)

    result = pd.concat(selected, ignore_index=True)

    # Shuffle without changing the sampled records.
    result = result.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return result


train_df = balanced_sample(
    train_df,
    TRAIN_IMAGES_PER_CLASS,
    "Training",
)

val_df = balanced_sample(
    val_df,
    VAL_IMAGES_PER_CLASS,
    "Validation",
)

print("\nDataset sampling complete.")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Number of classes:", NUM_CLASSES)

print("\nTraining samples per class:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation samples per class:")
print(val_df["label"].value_counts().sort_index())

# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

# EfficientNetB3 uses 300 x 300 input images.
# ViT-B/16 uses 224 x 224 input images.
#
# Both pretrained models use ImageNet normalization.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

efficientnet_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])

# ============================================================
# 5. DATASET CLASS
# ============================================================

class DriverDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = str(row["image_path"])
        label = int(row["label"])

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            efficientnet_image = efficientnet_transform(image)
            vit_image = vit_transform(image)

        return (
            efficientnet_image,
            vit_image,
            label,
        )


train_dataset = DriverDataset(train_df)
val_dataset = DriverDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# ============================================================
# 6. LOAD PRETRAINED MODELS
# ============================================================

print("\nLoading pretrained EfficientNetB3...")

efficientnet_weights = models.EfficientNet_B3_Weights.DEFAULT
efficientnet = models.efficientnet_b3(
    weights=efficientnet_weights
)

# Replace the classification layer with an identity layer.
# This returns the extracted feature vector.
efficientnet_feature_size = efficientnet.classifier[1].in_features
efficientnet.classifier = nn.Identity()

print("EfficientNetB3 feature size:", efficientnet_feature_size)

print("\nLoading pretrained Vision Transformer...")

vit_weights = models.ViT_B_16_Weights.DEFAULT
vit = models.vit_b_16(weights=vit_weights)

vit_feature_size = vit.heads.head.in_features
vit.heads = nn.Identity()

print("ViT feature size:", vit_feature_size)

# Freeze the two pretrained feature extractors.
for parameter in efficientnet.parameters():
    parameter.requires_grad = False

for parameter in vit.parameters():
    parameter.requires_grad = False

efficientnet = efficientnet.to(DEVICE)
vit = vit.to(DEVICE)

efficientnet.eval()
vit.eval()

# ============================================================
# 7. FEATURE FUSION CLASSIFIER
# ============================================================

class FeatureFusionClassifier(nn.Module):
    def __init__(
        self,
        efficientnet_size,
        vit_size,
        num_classes,
    ):
        super().__init__()

        combined_size = efficientnet_size + vit_size

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, num_classes),
        )

    def forward(self, efficientnet_features, vit_features):
        combined_features = torch.cat(
            [efficientnet_features, vit_features],
            dim=1,
        )

        return self.classifier(combined_features)


fusion_model = FeatureFusionClassifier(
    efficientnet_size=efficientnet_feature_size,
    vit_size=vit_feature_size,
    num_classes=NUM_CLASSES,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    fusion_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001,
)

# ============================================================
# 8. EXTRACT FROZEN FEATURES
# ============================================================

def extract_features(data_loader, description):
    efficientnet_features_list = []
    vit_features_list = []
    labels_list = []

    total_batches = len(data_loader)
    start_time = time.time()

    efficientnet.eval()
    vit.eval()

    with torch.no_grad():
        for batch_index, (
            efficientnet_images,
            vit_images,
            labels,
        ) in enumerate(data_loader, start=1):

            efficientnet_images = efficientnet_images.to(DEVICE)
            vit_images = vit_images.to(DEVICE)

            features_eff = efficientnet(efficientnet_images)
            features_vit = vit(vit_images)

            efficientnet_features_list.append(
                features_eff.cpu()
            )

            vit_features_list.append(
                features_vit.cpu()
            )

            labels_list.append(labels.cpu())

            if batch_index % 25 == 0 or batch_index == total_batches:
                print(
                    f"{description}: "
                    f"batch {batch_index}/{total_batches}"
                )

    elapsed = time.time() - start_time

    efficientnet_features = torch.cat(
        efficientnet_features_list, dim=0
    )

    vit_features = torch.cat(
        vit_features_list, dim=0
    )

    labels = torch.cat(labels_list, dim=0)

    print(
        f"{description} feature extraction finished "
        f"in {elapsed / 60:.2f} minutes."
    )

    return (
        efficientnet_features,
        vit_features,
        labels,
    )


print("\nExtracting training features...")
train_eff_features, train_vit_features, train_labels = (
    extract_features(train_loader, "Training")
)

print("\nExtracting validation features...")
val_eff_features, val_vit_features, val_labels = (
    extract_features(val_loader, "Validation")
)

# ============================================================
# 9. TRAIN THE FUSION CLASSIFIER
# ============================================================

train_eff_features = train_eff_features.to(DEVICE)
train_vit_features = train_vit_features.to(DEVICE)
train_labels = train_labels.to(DEVICE)

val_eff_features = val_eff_features.to(DEVICE)
val_vit_features = val_vit_features.to(DEVICE)
val_labels = val_labels.to(DEVICE)

best_val_accuracy = 0.0
best_epoch = 0
history = []

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "best_efficientnetb3_vit.pth",
)

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv",
)

start_training_time = time.time()

print("\nStarting fusion classifier training...")

for epoch in range(EPOCHS):
    fusion_model.train()

    epoch_loss = 0.0
    train_predictions = []
    train_targets = []

    # Shuffle feature indices each epoch.
    indices = torch.randperm(
        len(train_labels),
        device=DEVICE,
    )

    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]

        batch_eff = train_eff_features[batch_indices]
        batch_vit = train_vit_features[batch_indices]
        batch_labels = train_labels[batch_indices]

        optimizer.zero_grad()

        outputs = fusion_model(batch_eff, batch_vit)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_labels)

        predictions = torch.argmax(outputs, dim=1)

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_targets.extend(
            batch_labels.detach().cpu().numpy()
        )

    train_loss = epoch_loss / len(train_labels)

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions,
    )

    train_macro_f1 = f1_score(
        train_targets,
        train_predictions,
        average="macro",
        zero_division=0,
    )

    # Validation
    fusion_model.eval()

    with torch.no_grad():
        val_outputs = fusion_model(
            val_eff_features,
            val_vit_features,
        )

        val_predictions = torch.argmax(
            val_outputs, dim=1
        )

        val_loss = criterion(
            val_outputs,
            val_labels,
        ).item()

    val_predictions_np = val_predictions.cpu().numpy()
    val_targets_np = val_labels.cpu().numpy()

    val_accuracy = accuracy_score(
        val_targets_np,
        val_predictions_np,
    )

    val_macro_f1 = f1_score(
        val_targets_np,
        val_predictions_np,
        average="macro",
        zero_division=0,
    )

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        f"\nTrain Loss: {train_loss:.4f}"
        f"\nTrain Accuracy: {train_accuracy * 100:.2f}%"
        f"\nTrain Macro F1: {train_macro_f1:.4f}"
        f"\nValidation Loss: {val_loss:.4f}"
        f"\nValidation Accuracy: {val_accuracy * 100:.2f}%"
        f"\nValidation Macro F1: {val_macro_f1:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
    })

    # Save the best classifier using validation accuracy.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save({
            "classifier_state_dict": fusion_model.state_dict(),
            "efficientnet_feature_size": efficientnet_feature_size,
            "vit_feature_size": vit_feature_size,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_val_accuracy,
            "random_seed": RANDOM_SEED,
        }, checkpoint_path)

        print("Best checkpoint saved.")

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

# ============================================================
# 10. EVALUATE THE BEST CHECKPOINT
# ============================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)

fusion_model.load_state_dict(
    checkpoint["classifier_state_dict"]
)

fusion_model.eval()

with torch.no_grad():
    final_val_outputs = fusion_model(
        val_eff_features,
        val_vit_features,
    )

    final_predictions = torch.argmax(
        final_val_outputs, dim=1
    ).cpu().numpy()

final_targets = val_labels.cpu().numpy()

final_accuracy = accuracy_score(
    final_targets,
    final_predictions,
)

final_macro_f1 = f1_score(
    final_targets,
    final_predictions,
    average="macro",
    zero_division=0,
)

report = classification_report(
    final_targets,
    final_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

matrix = confusion_matrix(
    final_targets,
    final_predictions,
    labels=list(range(NUM_CLASSES)),
)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt",
)

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv",
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("EfficientNetB3 + ViT Classification Report\n")
    file.write("=" * 55 + "\n")
    file.write(f"Stream: {STREAM_NAME}\n")
    file.write(f"Training samples: {len(train_df)}\n")
    file.write(f"Validation samples: {len(val_df)}\n")
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Validation accuracy: {final_accuracy:.4f}\n")
    file.write(f"Validation macro F1: {final_macro_f1:.4f}\n\n")
    file.write(report)

pd.DataFrame(
    matrix,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(confusion_path)

total_training_minutes = (
    time.time() - start_training_time
) / 60

# ============================================================
# 11. FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FINAL RESULTS: EFFICIENTNETB3 + VIT")
print("=" * 65)

print(f"Best epoch: {best_epoch}")
print(f"Validation accuracy: {final_accuracy * 100:.2f}%")
print(f"Validation macro F1: {final_macro_f1:.4f}")
print(f"Classifier training time: {total_training_minutes:.2f} minutes")

print("\nClassification Report:\n")
print(report)

print("\nFiles saved:")
print("Best checkpoint:", checkpoint_path)
print("Training history:", history_path)
print("Classification report:", report_path)
print("Confusion matrix:", confusion_path)

print("\nExperiment completed.")
