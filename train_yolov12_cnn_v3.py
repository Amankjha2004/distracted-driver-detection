from pathlib import Path
import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO


# ============================================================
# YOLOv12 CLASSIFICATION V3 - FINE-TUNE THE V2 CHECKPOINT
# Uses the existing V2 sampled dataset and starts from V2's
# best checkpoint. V2 files are not deleted or overwritten.
# ============================================================

PROJECT_DIR = Path(r"D:\AI_FINAL_PROJECT")

V2_OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolov12_cnn_v2"
DATASET_DIR = V2_OUTPUT_DIR / "dataset_v2"
V2_CHECKPOINT = (
    V2_OUTPUT_DIR
    / "runs"
    / "yolov12n_classification_v2"
    / "weights"
    / "best.pt"
)

V3_OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolov12_cnn_v3"
RUNS_DIR = V3_OUTPUT_DIR / "runs"

IMAGE_SIZE = 224
EPOCHS = 15
BATCH_SIZE = 8
PATIENCE = 5
SEED = 42

# Keep these names aligned with the class-folder names in dataset_v2.
ACTIVITIES = [
    "safe_driving",
    "hair_makeup",
    "adjusting_radio",
    "gps_operating",
    "texting_right_hand",
    "texting_left_hand",
    "phone_right_hand",
    "phone_left_hand",
    "taking_picture",
    "talking_to_passenger",
    "singing_or_dancing",
    "fatigue_somnolence",
    "drinking_right_hand",
    "drinking_left_hand",
    "reaching_behind",
    "smoking",
]


def validate_inputs():
    """Fail early if the V2 checkpoint or prepared dataset is missing."""
    if not V2_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "The V2 best checkpoint was not found:\n"
            f"{V2_CHECKPOINT}\n\n"
            "Run the V2 training script first, or correct V2_CHECKPOINT."
        )

    for split_name in ("train", "val"):
        split_dir = DATASET_DIR / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Prepared V2 {split_name} folder was not found:\n"
                f"{split_dir}"
            )

        missing_classes = [
            name for name in ACTIVITIES
            if not (split_dir / name).is_dir()
        ]
        if missing_classes:
            raise FileNotFoundError(
                f"Missing class folders in {split_name}:\n"
                + ", ".join(missing_classes)
            )

    # Ensure the model is evaluated on the same set of class folders.
    train_classes = sorted(
        path.name for path in (DATASET_DIR / "train").iterdir()
        if path.is_dir()
    )
    val_classes = sorted(
        path.name for path in (DATASET_DIR / "val").iterdir()
        if path.is_dir()
    )
    if train_classes != val_classes:
        raise ValueError(
            "Train and validation class folders do not match.\n"
            f"Train: {train_classes}\nValidation: {val_classes}"
        )

    print("Input checks passed.")
    print(f"V2 checkpoint: {V2_CHECKPOINT}")
    print(f"Existing dataset: {DATASET_DIR}")
    print(f"V3 output folder: {V3_OUTPUT_DIR}")
    print("The V2 dataset and V2 results will not be deleted.")


def get_validation_files():
    """Collect validation image paths and their true folder labels."""
    paths = []
    true_labels = []

    for label in sorted(ACTIVITIES):
        class_dir = DATASET_DIR / "val" / label
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
            }:
                paths.append(path)
                true_labels.append(label)

    if not paths:
        raise RuntimeError(
            f"No validation images were found under {DATASET_DIR / 'val'}"
        )

    return paths, true_labels


def main():
    start_time = time.time()
    validate_inputs()
    V3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("YOLOv12 CLASSIFICATION V3 - CHECKPOINT FINE-TUNING")
    print("=" * 68)
    print("Device: CPU")
    print(f"Starting checkpoint: {V2_CHECKPOINT}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Maximum additional epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Early-stopping patience: {PATIENCE}")
    print("Learning rate: 0.0002")
    print("Horizontal flip: disabled")
    print("Vertical flip: disabled")

    # Loading best.pt initializes the model with the learned V2 weights.
    # Do not use resume=True here: that would resume a last.pt training state,
    # rather than fine-tuning the selected best.pt checkpoint.
    model = YOLO(str(V2_CHECKPOINT))

    print("\nStarting V3 fine-tuning...")
    train_results = model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        plots=True,
        project=str(RUNS_DIR),
        name="yolov12n_classification_v3",
        exist_ok=False,
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.1,
        weight_decay=0.0005,
        warmup_epochs=1,
        fliplr=0.0,
        flipud=0.0,
        degrees=3.0,
        translate=0.03,
        scale=0.05,
        hsv_h=0.005,
        hsv_s=0.2,
        hsv_v=0.15,
        verbose=True,
    )

    run_dir = Path(train_results.save_dir)
    best_checkpoint = run_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            "V3 training finished but its best checkpoint was not found:\n"
            f"{best_checkpoint}"
        )

    print(f"\nV3 best checkpoint: {best_checkpoint}")
    best_model = YOLO(str(best_checkpoint))

    print("\nEvaluating the V3 best checkpoint on the existing validation set...")
    metrics = best_model.val(
        data=str(DATASET_DIR),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,
        plots=True,
        project=str(RUNS_DIR),
        name="yolov12n_validation_v3",
        exist_ok=False,
    )

    validation_paths, true_labels = get_validation_files()
    print(f"\nGenerating predictions for {len(validation_paths)} validation images...")
    predictions = best_model.predict(
        source=[str(path) for path in validation_paths],
        imgsz=IMAGE_SIZE,
        device="cpu",
        batch=BATCH_SIZE,
        verbose=False,
        stream=False,
    )

    if len(predictions) != len(true_labels):
        raise RuntimeError(
            "Prediction count does not match validation image count: "
            f"{len(predictions)} predictions vs {len(true_labels)} labels."
        )

    predicted_labels = []
    for result in predictions:
        class_index = int(result.probs.top1)
        predicted_labels.append(result.names[class_index])

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=ACTIVITIES,
        target_names=ACTIVITIES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=ACTIVITIES,
    )

    report_path = V3_OUTPUT_DIR / "classification_report.txt"
    matrix_path = V3_OUTPUT_DIR / "confusion_matrix.csv"
    results_path = V3_OUTPUT_DIR / "final_results.txt"

    report_path.write_text(report_text, encoding="utf-8")
    pd.DataFrame(
        matrix,
        index=ACTIVITIES,
        columns=ACTIVITIES,
    ).to_csv(matrix_path, encoding="utf-8")

    with results_path.open("w", encoding="utf-8") as report:
        report.write("YOLOv12 Classification V3 - 3MDAD Day-RGB1\n")
        report.write("=" * 60 + "\n")
        report.write("Initialization: Fine-tuned from V2 best.pt\n")
        report.write(f"Starting checkpoint: {V2_CHECKPOINT}\n")
        report.write(f"Best V3 checkpoint: {best_checkpoint}\n")
        report.write(f"Dataset reused: {DATASET_DIR}\n")
        report.write(f"Maximum additional epochs: {EPOCHS}\n")
        report.write(f"Image size: {IMAGE_SIZE}\n")
        report.write(f"Elapsed minutes: {(time.time() - start_time) / 60:.2f}\n")
        report.write("\nUltralytics validation metrics:\n")
        for key, value in metrics.results_dict.items():
            report.write(f"{key}: {value}\n")
        report.write("\nPer-class classification report:\n")
        report.write(report_text)
        report.write("\n\nHeld-out test split was not used.\n")

    print("\n" + "=" * 68)
    print("YOLOv12 V3 EXPERIMENT COMPLETED")
    print("=" * 68)
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Results: {results_path}")
    print(f"Classification report: {report_path}")
    print(f"Confusion matrix: {matrix_path}")
    print(f"Elapsed minutes: {(time.time() - start_time) / 60:.2f}")
    print("The held-out test split was not used.")
    print("\nCompare V3 with V2 using top-1 accuracy and macro F1-score.")


if __name__ == "__main__":
    main()
