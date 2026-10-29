
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_DIR = Path(r"D:\AI_FINAL_PROJECT")
CSV_PATH = PROJECT_DIR / "3mdad_day_rgb1_subject_split_80_20.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "densenet121_svm_v3"

MODEL_PATH = OUTPUT_DIR / "best_densenet121_svm_v3.joblib"
MAPPING_PATH = OUTPUT_DIR / "class_mapping.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class TestImageDataset(Dataset):
    def __init__(self, frame, class_to_idx):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        with Image.open(Path(row["image_path"])) as image:
            image = TRANSFORM(image.convert("RGB"))

        label = self.class_to_idx[row["activity_label"]]
        return image, label


def build_feature_extractor(device):
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def extract_test_features(model, loader, device):
    features_list = []
    labels_list = []
    start = time.time()

    with torch.inference_mode():
        for batch_no, (images, labels) in enumerate(loader, start=1):
            features = model(images.to(device))
            features_list.append(features.cpu().numpy().astype(np.float32))
            labels_list.append(labels.numpy().astype(np.int64))

            if batch_no % 100 == 0 or batch_no == len(loader):
                print(f"Test batch {batch_no}/{len(loader)}")

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(
        f"Extracted {len(labels):,} test images in "
        f"{(time.time() - start) / 60:.1f} minutes."
    )
    return features, labels


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained V3 model not found. Run train_densenet121_svm_v3.py first."
        )

    if not CSV_PATH.is_file() or not MAPPING_PATH.is_file():
        raise FileNotFoundError("CSV or saved class mapping is missing.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    classes = mapping["classes"]
    class_to_idx = mapping["class_to_idx"]

    df = pd.read_csv(CSV_PATH)
    df = df[
        (df["stream"] == "Day-RGB1") &
        (df["new_split"] == "test")
    ].copy()

    if df.empty:
        raise ValueError("No held-out test images found.")

    if set(df["activity_label"].unique()) != set(classes):
        raise ValueError("Test split class labels do not match saved mapping.")

    # Confirm every test subject is excluded from train and validation.
    all_df = pd.read_csv(CSV_PATH)
    test_subjects = set(df["subject"].unique())
    train_subjects = set(
        all_df.loc[all_df["new_split"] == "train", "subject"].unique()
    )
    val_subjects = set(
        all_df.loc[all_df["new_split"] == "validation", "subject"].unique()
    )

    if test_subjects & (train_subjects | val_subjects):
        raise ValueError("Subject leakage detected in held-out test data.")

    print("Test images:", len(df))
    print("Held-out test subjects:", len(test_subjects))

    missing_paths = [
        p for p in df["image_path"].tolist() if not Path(p).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} test image paths are missing. "
            f"Example: {missing_paths[0]}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = TestImageDataset(df, class_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    backbone = build_feature_extractor(device)
    x_test, y_test = extract_test_features(backbone, loader, device)

    if x_test.shape[1] != 1024:
        raise ValueError(f"Expected 1024 features, got {x_test.shape[1]}.")

    # Test predictions happen only after model selection is complete.
    best_model = joblib.load(MODEL_PATH)
    predictions = best_model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(classes))),
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "test_classification_report.csv"
    )
    pd.DataFrame(
        matrix, index=classes, columns=classes
    ).to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

    results = {
        "evaluation": "held-out subject-wise test",
        "test_images": int(len(y_test)),
        "test_subjects": int(len(test_subjects)),
        "number_of_classes": len(classes),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "test_subjects_used_for_training": False,
        "test_subjects_used_for_model_selection": False,
    }

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL HELD-OUT TEST RESULTS")
    print(json.dumps(results, indent=2))
    print("\nSaved:")
    print(OUTPUT_DIR / "test_metrics.json")
    print(OUTPUT_DIR / "test_classification_report.csv")
    print(OUTPUT_DIR / "test_confusion_matrix.csv")


if __name__ == "__main__":
    main()
