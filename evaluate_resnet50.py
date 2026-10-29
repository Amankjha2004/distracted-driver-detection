
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# CONFIGURATION — MATCH THE TRAINING SCRIPT
# --------------------------------------------------

ROOT = Path(r"D:\AI_FINAL_PROJECT")

CSV_PATH = ROOT / "3mdad_master_metadata_split.csv"

CHECKPOINT_PATH = (
    ROOT / "outputs" / "resnet50_pretrained_baseline"
    / "best_resnet50.pth"
)

OUTPUT_DIR = ROOT / "outputs" / "resnet50_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STREAM = "Day-RGB1"
IMAGE_SIZE = 224
BATCH_SIZE = 8
VAL_PER_CLASS = 20
NUM_CLASSES = 16
SEED = 42

DEVICE = torch.device("cpu")

# Same preprocessing as training.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
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

        with Image.open(Path(row["image_path"])) as image:
            image = image.convert("RGB")
            image = transform(image)

        # IMPORTANT: Same mapping used during training.
        # activity_id 1 -> index 0, activity_id 16 -> index 15.
        label = int(row["activity_id"]) - 1

        return image, label


# --------------------------------------------------
# LOAD VALIDATION SAMPLES
# --------------------------------------------------

def load_validation_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {CSV_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required = {
        "image_path",
        "stream",
        "split",
        "activity_id",
        "activity_label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    # Match the training script's stream and validation split.
    subset = df[
        (df["stream"] == STREAM)
        & (df["split"] == "validation")
    ]

    selected = []

    # Match the original training sampling procedure exactly.
    for class_id in range(1, NUM_CLASSES + 1):
        class_rows = subset[subset["activity_id"] == class_id]

        if len(class_rows) < VAL_PER_CLASS:
            raise ValueError(
                f"Not enough validation images for activity ID {class_id}."
            )

        selected.append(
            class_rows.sample(
                n=VAL_PER_CLASS,
                random_state=SEED,
            )
        )

    validation_df = (
        pd.concat(selected)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    # Build class names in the same order as model output indices.
    class_names = []

    for class_id in range(1, NUM_CLASSES + 1):
        names = (
            validation_df.loc[
                validation_df["activity_id"] == class_id,
                "activity_label",
            ]
            .dropna()
            .unique()
        )

        if len(names) != 1:
            raise ValueError(
                f"Activity ID {class_id} does not map to exactly "
                f"one activity label: {names.tolist()}"
            )

        class_names.append(str(names[0]))

    # Verify all selected paths exist.
    missing_paths = [
        str(path)
        for path in validation_df["image_path"]
        if not Path(path).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} image files are missing. "
            f"Example: {missing_paths[0]}"
        )

    loader = DataLoader(
        DriverDataset(validation_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return loader, class_names, len(validation_df)


# --------------------------------------------------
# LOAD THE SAVED CHECKPOINT
# --------------------------------------------------

def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Check the saved experiment configuration.
    if checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError("Checkpoint class count does not match.")

    if checkpoint.get("image_size") != IMAGE_SIZE:
        raise ValueError("Checkpoint image size does not match.")

    if checkpoint.get("stream") != STREAM:
        raise ValueError("Checkpoint stream does not match.")

    if checkpoint.get("class_ids") != list(range(1, NUM_CLASSES + 1)):
        raise ValueError("Checkpoint activity ID mapping does not match.")

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

def main():
    print("=" * 60)
    print("RESNET50 VALIDATION EVALUATION — CORRECTED")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Stream:", STREAM)

    loader, class_names, total_images = load_validation_data()
    model = load_model()

    print("Validation images:", total_images)
    print("Classes:", len(class_names))
    print("\nModel output index -> activity label:")

    for index, name in enumerate(class_names):
        print(f"{index:2d} -> {name}")

    actual = []
    predicted = []

    print("\nRunning evaluation...")

    with torch.inference_mode():
        for batch_no, (images, labels) in enumerate(loader, start=1):
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.numpy().tolist())
            predicted.extend(predictions.cpu().numpy().tolist())

            processed = min(batch_no * BATCH_SIZE, total_images)
            print(
                f"Processed {processed}/{total_images} images",
                end="\r",
                flush=True,
            )

    y_true = np.array(actual)
    y_pred = np.array(predicted)

    accuracy = accuracy_score(y_true, y_pred)

    print("\n\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Correct predictions: {(y_true == y_pred).sum()}/{len(y_true)}")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Validation accuracy: {accuracy:.4%}\n\n")
        file.write(report_text)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Plot confusion matrix.
    fig, ax = plt.subplots(figsize=(15, 12))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    positions = np.arange(NUM_CLASSES)

    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(class_names, rotation=60, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("Actual activity")
    ax.set_title(f"ResNet50 Confusion Matrix — {STREAM}")

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nResults saved to:", OUTPUT_DIR)
    print(" - confusion_matrix.png")
    print(" - confusion_matrix.csv")
    print(" - classification_report.csv")
    print(" - classification_report.txt")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
