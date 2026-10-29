from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import joblib

# DenseNet121 feature extraction + SVM, 3MDAD Day-RGB1.
# Uses the prepared subject-wise V2 train/val folders; does not use test data.
PROJECT_DIR = Path(r"D:\AI_FINAL_PROJECT")
DATASET_DIR = PROJECT_DIR / "outputs" / "yolov12_cnn_v2" / "dataset_v2"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "densenet121_svm"
IMAGE_SIZE, BATCH_SIZE, SEED = 224, 16, 42
CLASSES = [
    "safe_driving", "hair_makeup", "adjusting_radio", "gps_operating",
    "texting_right_hand", "texting_left_hand", "phone_right_hand",
    "phone_left_hand", "taking_picture", "talking_to_passenger",
    "singing_or_dancing", "fatigue_somnolence", "drinking_right_hand",
    "drinking_left_hand", "reaching_behind", "smoking",
]


def make_loader(split):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(DATASET_DIR / split), transform=transform)
    if dataset.classes != sorted(CLASSES):
        raise ValueError(f"Unexpected class folders for {split}: {dataset.classes}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return dataset, loader


def extract_features(model, loader, cache_path, label):
    if cache_path.exists():
        print(f"Loading cached {label} features: {cache_path}")
        saved = np.load(cache_path, allow_pickle=False)
        return saved["features"], saved["labels"]
    features_list, labels_list = [], []
    model.eval()
    start = time.time()
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            feats = model(images)
            features_list.append(feats.cpu().numpy().astype(np.float32))
            labels_list.append(labels.numpy().astype(np.int64))
            if batch_idx % 25 == 0 or batch_idx == len(loader):
                print(f"{label}: batch {batch_idx}/{len(loader)}")
    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)
    np.savez_compressed(cache_path, features=features, labels=labels)
    print(f"Extracted {len(labels)} {label} images in {(time.time()-start)/60:.1f} min")
    return features, labels


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.time()
    for split in ("train", "val"):
        if not (DATASET_DIR / split).is_dir():
            raise FileNotFoundError(f"Missing {split} dataset folder: {DATASET_DIR / split}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("DenseNet121 + SVM | 3MDAD Day-RGB1 | CPU")
    print("Using prepared V2 dataset folders; held-out test data stays untouched.")

    train_ds, train_loader = make_loader("train")
    val_ds, val_loader = make_loader("val")
    weights = DenseNet121_Weights.DEFAULT  # Downloads ImageNet weights if not cached.
    backbone = models.densenet121(weights=weights)
    backbone.classifier = nn.Identity()  # 1024-dimensional feature vector
    backbone.eval()

    x_train, y_train = extract_features(backbone, train_loader, OUTPUT_DIR / "train_features.npz", "train")
    x_val, y_val = extract_features(backbone, val_loader, OUTPUT_DIR / "val_features.npz", "validation")
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train and validation class ordering differs.")

    candidates = {
        "linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("svm", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=10000)),
        ]),
        "rbf_svm_c1": Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=1000)),
        ]),
    }
    comparisons = []
    best_name, best_model, best_macro_f1 = None, None, -1.0
    for name, clf in candidates.items():
        print(f"\nTraining {name}...")
        fit_start = time.time()
        clf.fit(x_train, y_train)
        elapsed_fit = time.time() - fit_start
        pred = clf.predict(x_val)
        acc = accuracy_score(y_val, pred)
        macro = f1_score(y_val, pred, average="macro", zero_division=0)
        weighted = f1_score(y_val, pred, average="weighted", zero_division=0)
        comparisons.append({"model": name, "validation_accuracy": acc, "macro_f1": macro,
                            "weighted_f1": weighted, "fit_seconds": round(elapsed_fit, 2)})
        print(f"{name}: accuracy={acc:.4f}, macro-F1={macro:.4f}, weighted-F1={weighted:.4f}")
        if macro > best_macro_f1:
            best_name, best_model, best_macro_f1 = name, clf, macro

    pred = best_model.predict(x_val)
    report = classification_report(
        y_val, pred, labels=list(range(len(train_ds.classes))),
        target_names=train_ds.classes, digits=4, zero_division=0
    )
    matrix = confusion_matrix(y_val, pred, labels=list(range(len(train_ds.classes))))
    model_path = OUTPUT_DIR / "best_densenet121_svm.joblib"
    joblib.dump(best_model, model_path)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(matrix, index=train_ds.classes, columns=train_ds.classes).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    pd.DataFrame(comparisons).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    summary = {
        "experiment": "DenseNet121 + SVM", "dataset": "3MDAD Day-RGB1",
        "backbone": "ImageNet-pretrained DenseNet121", "train_images": int(len(y_train)),
        "validation_images": int(len(y_val)), "feature_dimension": int(x_train.shape[1]),
        "selected_svm": best_name, "validation_accuracy": float(accuracy_score(y_val, pred)),
        "validation_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
        "elapsed_minutes": round((time.time()-start)/60, 2), "classifier_path": str(model_path),
        "held_out_test_used": False,
    }
    (OUTPUT_DIR / "final_results.txt").write_text(json.dumps(summary, indent=2) + "\n\nPer-class report:\n" + report, encoding="utf-8")
    print("\nDENSENET121 + SVM COMPLETED")
    print(json.dumps(summary, indent=2))
    print(f"Results: {OUTPUT_DIR / 'final_results.txt'}")
    print(f"Comparison: {OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.csv'}")

if __name__ == "__main__":
    main()
