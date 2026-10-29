
from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights

from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = Path(r"D:\AI_FINAL_PROJECT")
CSV_PATH = PROJECT_DIR / "3mdad_day_rgb1_subject_split_80_20.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "densenet121_svm_v3"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

# RBF SVM can be very slow on tens of thousands of images.
# It is trained on a deterministic, stratified subset.
RBF_MAX_SAMPLES = 15000

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET
# ============================================================
class RGBMetadataDataset(Dataset):
    def __init__(self, frame, class_to_idx):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = TRANSFORM(image)

        label = self.class_to_idx[row["activity_label"]]
        return image, label


def load_and_validate_metadata():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {
        "image_path", "activity_label", "subject", "stream", "new_split"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df = df[df["stream"] == "Day-RGB1"].copy()

    if df.empty:
        raise ValueError("No Day-RGB1 records found.")

    if df[["image_path", "activity_label", "subject", "new_split"]].isna().any().any():
        raise ValueError("Missing values found in required metadata columns.")

    allowed = {"train", "validation", "test"}
    if set(df["new_split"].unique()) != allowed:
        raise ValueError("Expected train, validation, and test splits.")

    # Ensure every subject belongs to exactly one split.
    subject_split_counts = df.groupby("subject")["new_split"].nunique()
    if (subject_split_counts != 1).any():
        raise ValueError("Subject leakage detected: a subject appears in multiple splits.")

    # Ensure all splits use the same 16 classes.
    classes = sorted(df["activity_label"].unique().tolist())
    if len(classes) != 16:
        raise ValueError(f"Expected 16 classes; found {len(classes)}.")

    for split in ["train", "validation", "test"]:
        part = df[df["new_split"] == split]
        if set(part["activity_label"].unique()) != set(classes):
            raise ValueError(f"{split} does not contain all 16 classes.")

        # Check that referenced image files actually exist.
        missing_paths = [
            p for p in part["image_path"].head(100).tolist()
            if not Path(p).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Some {split} image paths are invalid, e.g. {missing_paths[0]}"
            )

        print(
            f"{split}: {len(part):,} images, "
            f"{part['subject'].nunique()} subjects"
        )

    return df, classes


# ============================================================
# DENSENET121 FEATURE EXTRACTION
# ============================================================
def build_feature_extractor(device):
    weights = DenseNet121_Weights.DEFAULT
    backbone = models.densenet121(weights=weights)
    backbone.classifier = torch.nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def extract_densenet_features(model, frame, class_to_idx, cache_path, device):
    # Cache is unique to this V3 experiment.
    if cache_path.is_file():
        print(f"Loading cached features: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as saved:
            return saved["features"], saved["labels"]

    dataset = RGBMetadataDataset(frame, class_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    all_features = []
    all_labels = []
    start = time.time()

    with torch.inference_mode():
        for batch_no, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            features = model(images)

            all_features.append(features.cpu().numpy().astype(np.float32))
            all_labels.append(labels.numpy().astype(np.int64))

            if batch_no % 100 == 0 or batch_no == len(loader):
                print(
                    f"{cache_path.stem}: batch {batch_no}/{len(loader)}"
                )

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
    )

    print(
        f"Extracted {len(labels):,} images in "
        f"{(time.time() - start) / 60:.1f} minutes."
    )
    return features, labels


# ============================================================
# SVM TRAINING AND VALIDATION
# ============================================================
def train_svm(name, classifier, x_train, y_train, x_val, y_val):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", classifier),
    ])

    print(f"\nTraining {name}...")
    start = time.time()
    pipeline.fit(x_train, y_train)

    val_predictions = pipeline.predict(x_val)
    metrics = {
        "model": name,
        "validation_accuracy": float(
            accuracy_score(y_val, val_predictions)
        ),
        "validation_macro_f1": float(
            f1_score(y_val, val_predictions, average="macro", zero_division=0)
        ),
        "validation_weighted_f1": float(
            f1_score(y_val, val_predictions, average="weighted", zero_division=0)
        ),
        "training_seconds": round(time.time() - start, 2),
    }

    print(json.dumps(metrics, indent=2))
    return pipeline, metrics


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df, classes = load_and_validate_metadata()
    class_to_idx = {name: index for index, name in enumerate(classes)}

    train_df = df[df["new_split"] == "train"].copy()
    val_df = df[df["new_split"] == "validation"].copy()

    # Test records are deliberately not loaded or used in this script.
    backbone = build_feature_extractor(device)

    x_train, y_train = extract_densenet_features(
        backbone, train_df, class_to_idx,
        OUTPUT_DIR / "train_features.npz", device
    )
    x_val, y_val = extract_densenet_features(
        backbone, val_df, class_to_idx,
        OUTPUT_DIR / "validation_features.npz", device
    )

    if x_train.shape[1] != 1024:
        raise ValueError(f"Expected 1024 features, got {x_train.shape[1]}.")

    if not np.array_equal(y_train, np.array([
        class_to_idx[label] for label in train_df["activity_label"]
    ])):
        raise ValueError("Training feature cache label order mismatch.")

    if not np.array_equal(y_val, np.array([
        class_to_idx[label] for label in val_df["activity_label"]
    ])):
        raise ValueError("Validation feature cache label order mismatch.")

    candidates = []

    linear_model, linear_metrics = train_svm(
        "LinearSVC",
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=10000,
        ),
        x_train, y_train, x_val, y_val,
    )
    candidates.append((linear_model, linear_metrics))

    # Fit RBF SVM on a reproducible stratified subset to limit runtime.
    if len(y_train) > RBF_MAX_SAMPLES:
        x_rbf, _, y_rbf, _ = train_test_split(
            x_train,
            y_train,
            train_size=RBF_MAX_SAMPLES,
            stratify=y_train,
            random_state=SEED,
        )
    else:
        x_rbf, y_rbf = x_train, y_train

    rbf_model, rbf_metrics = train_svm(
        "RBF_SVC",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            cache_size=1000,
        ),
        x_rbf, y_rbf, x_val, y_val,
    )
    candidates.append((rbf_model, rbf_metrics))

    # Select solely by validation macro-F1.
    best_model, best_metrics = max(
        candidates,
        key=lambda item: item[1]["validation_macro_f1"],
    )
    best_name = best_metrics["model"]

    model_path = OUTPUT_DIR / "best_densenet121_svm_v3.joblib"
    joblib.dump(best_model, model_path)

    # Save class order so the separate test script uses the same mapping.
    with open(OUTPUT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {"classes": classes, "class_to_idx": class_to_idx},
            f,
            indent=2,
        )

    pd.DataFrame([item[1] for item in candidates]).to_csv(
        OUTPUT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    summary = {
        "backbone": "DenseNet121 ImageNet pretrained",
        "feature_dimension": int(x_train.shape[1]),
        "train_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "test_used_for_training_or_selection": False,
        "selected_svm": best_name,
        "selected_validation_metrics": best_metrics,
        "model_path": str(model_path),
        "classes": classes,
    }
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining finished.")
    print("Selected model:", best_name)
    print("Saved model:", model_path)
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
