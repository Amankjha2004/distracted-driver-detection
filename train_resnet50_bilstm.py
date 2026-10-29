
import os
import re
import time
import random
from pathlib import Path

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
    PROJECT_DIR, "outputs", "resnet50_bilstm_improved"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STREAM_NAME = "Day-RGB1"
NUM_CLASSES = 16
SEQUENCE_LENGTH = 8

TRAIN_SEQUENCES_PER_CLASS = 25
VAL_SEQUENCES_PER_CLASS = 8

BATCH_SIZE = 8
MAX_EPOCHS = 20
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01

LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
DROPOUT = 0.50

EARLY_STOPPING_PATIENCE = 4
LR_PATIENCE = 2
RANDOM_SEED = 42

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

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("IMPROVED RESNET50 + BILSTM")
print("=" * 70)
print("Device:", DEVICE)
print("Metadata:", CSV_PATH)
print("Output:", OUTPUT_DIR)

# ============================================================
# 2. LOAD METADATA
# ============================================================

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"Metadata CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image_path",
    "stream",
    "split",
    "subject",
    "activity_id",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required CSV columns: {sorted(missing)}"
    )

df["image_path"] = df["image_path"].astype(str).str.strip()
df["stream"] = df["stream"].astype(str).str.strip()
df["split"] = (
    df["split"].astype(str).str.strip().str.lower()
)
df["subject"] = df["subject"].astype(str).str.strip()

df["activity_id"] = pd.to_numeric(
    df["activity_id"], errors="coerce"
)

df = df.dropna(
    subset=[
        "image_path",
        "stream",
        "split",
        "subject",
        "activity_id",
    ]
).copy()

df["activity_id"] = df["activity_id"].astype(int)

df["split"] = df["split"].replace({
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "testing": "test",
})

df = df[
    (df["stream"] == STREAM_NAME)
    & df["activity_id"].between(1, NUM_CLASSES)
].copy()

if df.empty:
    raise ValueError(
        f"No records found for stream {STREAM_NAME}."
    )

df["label"] = df["activity_id"] - 1

# ============================================================
# 3. CREATE RECORDING AND FRAME COLUMNS FIRST
# ============================================================

def get_frame_number(image_path):
    filename = Path(str(image_path)).stem

    matches = re.findall(
        r"F(\d+)",
        filename,
        flags=re.IGNORECASE,
    )

    return int(matches[-1]) if matches else -1


# IMPORTANT: create these columns BEFORE train_df/val_df.
df["recording_key"] = df["image_path"].map(
    lambda p: str(Path(str(p)).parent)
)

df["frame_number"] = df["image_path"].map(
    get_frame_number
)

print("\nCreated required columns:")
print("recording_key:", "recording_key" in df.columns)
print("frame_number:", "frame_number" in df.columns)

# ============================================================
# 4. CHECK RECORDING CONSISTENCY
# ============================================================

for recording_key, group in df.groupby(
    "recording_key",
    sort=True,
):
    if group["subject"].nunique() != 1:
        raise ValueError(
            f"Multiple subjects in recording: {recording_key}"
        )

    if group["activity_id"].nunique() != 1:
        raise ValueError(
            f"Multiple activity labels in recording: {recording_key}"
        )

    if group["split"].nunique() != 1:
        raise ValueError(
            f"Recording occurs in multiple splits: {recording_key}"
        )

print("Recording consistency check passed.")

# NOW create the subsets. Both inherit the new columns.
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

if train_df.empty or val_df.empty:
    raise ValueError(
        "Training or validation data is empty."
    )

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print(
    "Training recordings:",
    train_df["recording_key"].nunique(),
)
print(
    "Validation recordings:",
    val_df["recording_key"].nunique(),
)

# ============================================================
# 5. SUBJECT-LEVEL SPLIT AUDIT
# ============================================================

train_subjects = set(train_df["subject"].unique())
val_subjects = set(val_df["subject"].unique())

test_subjects = set(
    df.loc[df["split"] == "test", "subject"].unique()
)

if train_subjects & val_subjects:
    raise ValueError("Training/validation subject leakage detected.")

if train_subjects & test_subjects:
    raise ValueError("Training/test subject leakage detected.")

if val_subjects & test_subjects:
    raise ValueError("Validation/test subject leakage detected.")

print("\nSubject split verification passed.")
print("Training subjects:", len(train_subjects))
print("Validation subjects:", len(val_subjects))
print("Test subjects:", len(test_subjects))

# ============================================================
# 6. BUILD CONSECUTIVE TEMPORAL SEQUENCES
# ============================================================

def build_sequences(dataframe, sequences_per_class, split_name):
    required = {
        "image_path",
        "subject",
        "label",
        "recording_key",
        "frame_number",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{split_name} DataFrame is missing columns: "
            f"{sorted(missing)}"
        )

    candidates = []

    for recording_key, group in dataframe.groupby(
        "recording_key",
        sort=True,
    ):
        group = group.sort_values(
            ["frame_number", "image_path"]
        ).reset_index(drop=True)

        if len(group) < SEQUENCE_LENGTH:
            continue

        frame_numbers = group["frame_number"].to_numpy()
        valid_starts = []

        for start in range(
            len(group) - SEQUENCE_LENGTH + 1
        ):
            numbers = frame_numbers[
                start:start + SEQUENCE_LENGTH
            ]

            if np.any(numbers < 0):
                continue

            if not np.all(np.diff(numbers) == 1):
                continue

            valid_starts.append(start)

        if not valid_starts:
            continue

        rng = random.Random(
            f"{RANDOM_SEED}_{split_name}_{recording_key}"
        )
        chosen_start = rng.choice(valid_starts)

        window = group.iloc[
            chosen_start:
            chosen_start + SEQUENCE_LENGTH
        ]

        candidates.append({
            "recording_key": recording_key,
            "subject": str(window.iloc[0]["subject"]),
            "label": int(window.iloc[0]["label"]),
            "paths": window["image_path"].tolist(),
            "first_frame": int(window.iloc[0]["frame_number"]),
            "last_frame": int(window.iloc[-1]["frame_number"]),
        })

    if not candidates:
        raise ValueError(
            f"No consecutive sequences found for {split_name}."
        )

    candidates_df = pd.DataFrame(candidates)
    selected = []

    for label in range(NUM_CLASSES):
        class_candidates = candidates_df[
            candidates_df["label"] == label
        ].copy()

        # Select no more than one recording per subject
        # for each activity in this pilot.
        class_candidates = class_candidates.drop_duplicates(
            subset=["subject"]
        )

        class_candidates = class_candidates.sample(
            frac=1,
            random_state=RANDOM_SEED,
        )

        if len(class_candidates) < sequences_per_class:
            raise ValueError(
                f"{split_name}, class {label + 1}: "
                f"only {len(class_candidates)} usable "
                f"subject-distinct sequences; "
                f"{sequences_per_class} required."
            )

        selected.append(
            class_candidates.head(sequences_per_class)
        )

    result = pd.concat(
        selected,
        ignore_index=True,
    )

    result = result.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return result


print("\nBuilding training sequences...")

train_sequences = build_sequences(
    train_df,
    TRAIN_SEQUENCES_PER_CLASS,
    "train",
)

print("\nBuilding validation sequences...")

val_sequences = build_sequences(
    val_df,
    VAL_SEQUENCES_PER_CLASS,
    "validation",
)

print("\nTraining sequences:", len(train_sequences))
print("Validation sequences:", len(val_sequences))

print("\nTraining sequences per class:")
print(train_sequences["label"].value_counts().sort_index())

print("\nValidation sequences per class:")
print(val_sequences["label"].value_counts().sort_index())

# Save manifests for reproducibility.
train_sequences.drop(columns=["paths"]).to_csv(
    os.path.join(OUTPUT_DIR, "train_sequence_manifest.csv"),
    index=False,
)

val_sequences.drop(columns=["paths"]).to_csv(
    os.path.join(OUTPUT_DIR, "validation_sequence_manifest.csv"),
    index=False,
)

# ============================================================
# 7. IMAGE TRANSFORMS AND DATASET
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class TemporalDriverDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences.reset_index(drop=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        row = self.sequences.iloc[index]
        frames = []

        for image_path in row["paths"]:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Image not found:\n{image_path}"
                )

            with Image.open(image_path) as image:
                frames.append(
                    image_transform(image.convert("RGB"))
                )

        return torch.stack(frames), int(row["label"])


train_dataset = TemporalDriverDataset(train_sequences)
val_dataset = TemporalDriverDataset(val_sequences)

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
# 8. LOAD PRETRAINED RESNET50
# ============================================================

print("\nLoading pretrained ResNet50...")

resnet = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

resnet_feature_size = resnet.fc.in_features
resnet.fc = nn.Identity()

for parameter in resnet.parameters():
    parameter.requires_grad = False

resnet = resnet.to(DEVICE)
resnet.eval()

print("ResNet50 feature size:", resnet_feature_size)

# ============================================================
# 9. EXTRACT FEATURES
# ============================================================

def extract_features(loader, description):
    feature_batches = []
    label_batches = []
    start_time = time.time()

    resnet.eval()

    with torch.no_grad():
        for batch_number, (frames, labels) in enumerate(
            loader,
            start=1,
        ):
            batch_size = frames.shape[0]

            frames = frames.reshape(
                batch_size * SEQUENCE_LENGTH,
                3,
                224,
                224,
            ).to(DEVICE)

            features = resnet(frames)

            features = features.reshape(
                batch_size,
                SEQUENCE_LENGTH,
                resnet_feature_size,
            )

            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())

            if (
                batch_number % 5 == 0
                or batch_number == len(loader)
            ):
                print(
                    f"{description}: "
                    f"{batch_number}/{len(loader)} batches"
                )

    print(
        f"{description} extraction time: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )

    return (
        torch.cat(feature_batches, dim=0).to(DEVICE),
        torch.cat(label_batches, dim=0).to(DEVICE),
    )


print("\nExtracting training features...")
train_features, train_labels = extract_features(
    train_loader, "Training"
)

print("\nExtracting validation features...")
val_features, val_labels = extract_features(
    val_loader, "Validation"
)

# ============================================================
# 10. DEFINE RESNET50 + BILSTM
# ============================================================

class ResNet50BiLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
    ):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, sequence_features):
        lstm_output, _ = self.bilstm(sequence_features)

        # Aggregate information from every time step.
        pooled_features = lstm_output.mean(dim=1)

        return self.classifier(pooled_features)


model = ResNet50BiLSTM(
    input_size=resnet_feature_size,
    hidden_size=LSTM_HIDDEN_SIZE,
    num_layers=LSTM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=LR_PATIENCE,
)

# ============================================================
# 11. TRAIN WITH EARLY STOPPING
# ============================================================

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "best_resnet50_bilstm.pth",
)

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv",
)

best_val_accuracy = -1.0
best_epoch = 0
epochs_without_improvement = 0
history = []

print("\nStarting BiLSTM training...")

training_start = time.time()

for epoch in range(MAX_EPOCHS):
    model.train()

    total_loss = 0.0
    train_predictions = []
    train_targets = []

    indices = torch.randperm(
        len(train_labels),
        device=DEVICE,
    )

    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]

        batch_features = train_features[batch_indices]
        batch_labels = train_labels[batch_indices]

        optimizer.zero_grad()

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item() * len(batch_labels)

        predictions = outputs.argmax(dim=1)

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        train_targets.extend(
            batch_labels.detach().cpu().numpy()
        )

    train_loss = total_loss / len(train_labels)
    train_accuracy = accuracy_score(
        train_targets, train_predictions
    )
    train_macro_f1 = f1_score(
        train_targets,
        train_predictions,
        average="macro",
        zero_division=0,
    )

    model.eval()

    with torch.no_grad():
        val_outputs = model(val_features)
        val_loss = criterion(val_outputs, val_labels).item()
        val_predictions = val_outputs.argmax(dim=1)

    val_targets_np = val_labels.cpu().numpy()
    val_predictions_np = val_predictions.cpu().numpy()

    val_accuracy = accuracy_score(
        val_targets_np, val_predictions_np
    )
    val_macro_f1 = f1_score(
        val_targets_np,
        val_predictions_np,
        average="macro",
        zero_division=0,
    )

    scheduler.step(val_accuracy)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch [{epoch + 1}/{MAX_EPOCHS}]"
        f"\nLearning rate: {current_lr:.7f}"
        f"\nTrain loss: {train_loss:.4f}"
        f"\nTrain accuracy: {train_accuracy * 100:.2f}%"
        f"\nTrain macro F1: {train_macro_f1:.4f}"
        f"\nValidation loss: {val_loss:.4f}"
        f"\nValidation accuracy: {val_accuracy * 100:.2f}%"
        f"\nValidation macro F1: {val_macro_f1:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "learning_rate": current_lr,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
    })

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "resnet_feature_size": resnet_feature_size,
            "hidden_size": LSTM_HIDDEN_SIZE,
            "num_layers": LSTM_LAYERS,
            "num_classes": NUM_CLASSES,
            "sequence_length": SEQUENCE_LENGTH,
            "class_names": CLASS_NAMES,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_val_accuracy,
            "random_seed": RANDOM_SEED,
        }, checkpoint_path)

        print("Best checkpoint saved.")
    else:
        epochs_without_improvement += 1

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")
        break

# ============================================================
# 12. FINAL VALIDATION EVALUATION
# ============================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    final_outputs = model(val_features)
    final_predictions = final_outputs.argmax(dim=1).cpu().numpy()

final_targets = val_labels.cpu().numpy()

final_accuracy = accuracy_score(
    final_targets, final_predictions
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
    OUTPUT_DIR, "classification_report.txt"
)
confusion_path = os.path.join(
    OUTPUT_DIR, "confusion_matrix.csv"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Improved ResNet50 + BiLSTM\n")
    file.write("=" * 45 + "\n")
    file.write(f"Stream: {STREAM_NAME}\n")
    file.write(f"Sequence length: {SEQUENCE_LENGTH}\n")
    file.write(f"Training sequences: {len(train_sequences)}\n")
    file.write(f"Validation sequences: {len(val_sequences)}\n")
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Validation accuracy: {final_accuracy:.4f}\n")
    file.write(f"Validation macro F1: {final_macro_f1:.4f}\n\n")
    file.write(report)

pd.DataFrame(
    matrix,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(confusion_path)

print("\n" + "=" * 70)
print("FINAL RESULTS: IMPROVED RESNET50 + BILSTM")
print("=" * 70)
print("Best epoch:", best_epoch)
print(f"Validation accuracy: {final_accuracy * 100:.2f}%")
print(f"Validation macro F1: {final_macro_f1:.4f}")
print(f"Training time after feature extraction: {(time.time() - training_start) / 60:.2f} minutes")

print("\nClassification report:\n")
print(report)

print("\nSaved files:")
print("Checkpoint:", checkpoint_path)
print("Training history:", history_path)
print("Classification report:", report_path)
print("Confusion matrix:", confusion_path)

print("\nExperiment completed.")
print("Test data was not used for training or model selection.")
