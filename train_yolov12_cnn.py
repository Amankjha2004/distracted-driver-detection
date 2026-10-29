
from pathlib import Path
import random
import re
import shutil
import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# YOLOv12 CLASSIFICATION V2 - 3MDAD DAY-RGB1
# More frames per recording, subject-wise split, no horizontal
# flip, longer training, and detailed validation metrics.
# ============================================================

PROJECT_DIR = Path(r"D:\AI_FINAL_PROJECT")
CSV_PATH = PROJECT_DIR / "3mdad_master_metadata_split.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolov12_cnn_v2"
DATASET_DIR = OUTPUT_DIR / "dataset_v2"

MODEL_CONFIG = "yolo12n-cls.yaml"

IMAGE_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 8
FRAMES_PER_RECORDING = 8
PATIENCE = 8
SEED = 42

random.seed(SEED)

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


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Missing {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_label(value):
    return str(value).strip().lower().replace(" ", "_")


def resolve_image_path(value):
    path = Path(str(value))

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


def frame_number(path):
    """Extract a frame number such as F123 from a filename."""
    match = re.search(r"F(\d+)", path.stem, flags=re.IGNORECASE)

    if match:
        return int(match.group(1))

    return None


def evenly_spaced_frames(group, path_col, limit):
    """
    Select frames spread across one recording.
    This reduces the chance of selecting only adjacent frames.
    """
    group = group.copy()

    group["_resolved_path"] = group[path_col].apply(
        resolve_image_path
    )

    group = group[
        group["_resolved_path"].apply(
            lambda path: path.is_file()
        )
    ].copy()

    if group.empty:
        return group

    group["_frame_number"] = group[
        "_resolved_path"
    ].apply(frame_number)

    if group["_frame_number"].notna().all():
        group = group.sort_values("_frame_number")
    else:
        group = group.sort_values(
            "_resolved_path",
            key=lambda series: series.astype(str).str.lower()
        )

    group = group.drop_duplicates(
        subset=["_resolved_path"]
    )

    if len(group) <= limit:
        return group

    # Choose approximately evenly spaced positions, including
    # the start and end of the recording.
    positions = [
        round(i * (len(group) - 1) / (limit - 1))
        for i in range(limit)
    ] if limit > 1 else [len(group) // 2]

    return group.iloc[positions].copy()


def prepare_split(source_df, path_col, label_col,
                  split_name):
    print(f"\nPreparing {split_name} dataset...")

    split_dir = DATASET_DIR / split_name
    counts = {}

    # Each parent directory represents one recording in the
    # inspected Day-RGB1 metadata.
    source_df = source_df.copy()
    source_df["_recording_key"] = source_df[
        path_col
    ].apply(
        lambda value: str(
            resolve_image_path(value).parent
        ).lower()
    )

    for label in ACTIVITIES:
        class_df = source_df[
            source_df[label_col] == label
        ].copy()

        destination = split_dir / label
        destination.mkdir(parents=True, exist_ok=True)

        copied = 0

        recording_groups = list(
            class_df.groupby("_recording_key", sort=True)
        )

        for _, recording_df in recording_groups:
            selected = evenly_spaced_frames(
                recording_df,
                path_col,
                FRAMES_PER_RECORDING,
            )

            for _, row in selected.iterrows():
                source = row["_resolved_path"]

                target = destination / (
                    f"{source.parent.name}_"
                    f"{source.stem}{source.suffix.lower()}"
                )

                # The source filename and recording name together
                # make collisions unlikely; add a suffix if needed.
                suffix = 1
                original_target = target

                while target.exists():
                    target = original_target.with_name(
                        f"{original_target.stem}_{suffix}"
                        f"{original_target.suffix}"
                    )
                    suffix += 1

                shutil.copy2(source, target)
                copied += 1

        counts[label] = copied
        print(
            f"{label}: {copied} images from "
            f"{len(recording_groups)} recordings"
        )

        if copied == 0:
            raise RuntimeError(
                f"No images prepared for class '{label}' "
                f"in split '{split_name}'."
            )

    return counts


def main():
    start_time = time.time()

    print("=" * 68)
    print("YOLOv12 CLASSIFICATION - IMPROVED EXPERIMENT V2")
    print("=" * 68)
    print("Device: CPU")
    print(f"Metadata: {CSV_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Frames per recording: {FRAMES_PER_RECORDING}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Early-stopping patience: {PATIENCE}")

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    path_col = find_column(
        df, ["image_path", "path"], "image path"
    )
    split_col = find_column(
        df, ["split", "dataset_split", "set"], "split"
    )
    label_col = find_column(
        df,
        ["activity_label", "label", "classname"],
        "activity label",
    )
    stream_col = find_column(
        df, ["stream", "camera"], "stream"
    )

    df[split_col] = (
        df[split_col].astype(str).str.lower().str.strip()
    )
    df[label_col] = df[label_col].apply(normalize_label)

    df = df[
        df[stream_col].astype(str).str.strip().eq("Day-RGB1")
    ].copy()

    train_df = df[
        df[split_col].isin({"train", "training"})
        & df[label_col].isin(ACTIVITIES)
    ].copy()

    val_df = df[
        df[split_col].isin({"val", "valid", "validation"})
        & df[label_col].isin(ACTIVITIES)
    ].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Training or validation metadata is empty. "
            "Check the split, stream and labels."
        )

    if "subject" not in df.columns:
        raise ValueError(
            "The subject column is missing. "
            "Cannot verify subject-wise separation."
        )

    train_subjects = set(train_df["subject"].astype(str))
    val_subjects = set(val_df["subject"].astype(str))

    overlap = train_subjects.intersection(val_subjects)

    if overlap:
        raise ValueError(
            f"Subject leakage detected: {sorted(overlap)}"
        )

    print("\nSubject separation check passed.")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Training metadata rows: {len(train_df)}")
    print(f"Validation metadata rows: {len(val_df)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only this new V2 generated dataset is removed/rebuilt.
    # The original dataset and V1 results remain untouched.
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    train_counts = prepare_split(
        train_df, path_col, label_col, "train"
    )

    val_counts = prepare_split(
        val_df, path_col, label_col, "val"
    )

    counts_path = OUTPUT_DIR / "dataset_sample_counts.csv"

    pd.DataFrame({
        "class": ACTIVITIES,
        "train_images": [
            train_counts[label] for label in ACTIVITIES
        ],
        "validation_images": [
            val_counts[label] for label in ACTIVITIES
        ],
    }).to_csv(counts_path, index=False)

    print("\nPrepared dataset totals:")
    print(f"Training images: {sum(train_counts.values())}")
    print(f"Validation images: {sum(val_counts.values())}")
    print(f"Sample report: {counts_path}")

    # Verify dependencies before model initialization.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed in this environment.\n"
            "Run:\n"
            r".\.venv\Scripts\python.exe -m pip install ultralytics"
        ) from exc

    print(f"\nInitializing architecture: {MODEL_CONFIG}")
    print("Training from scratch; no pretrained classification weights.")

    model = YOLO(MODEL_CONFIG)

    print("\nStarting training...")
    results = model.train(
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
        project=str(OUTPUT_DIR / "runs"),
        name="yolov12n_classification_v2",
        exist_ok=True,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best_checkpoint = run_dir / "weights" / "best.pt"

    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            "The best checkpoint was not found:\n"
            f"{best_checkpoint}\n"
            "Review the training logs."
        )

    print(f"\nBest checkpoint: {best_checkpoint}")

    best_model = YOLO(str(best_checkpoint))

    print("\nRunning validation metrics...")
    metrics = best_model.val(
        data=str(DATASET_DIR),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR / "runs"),
        name="yolov12n_validation_v2",
        exist_ok=True,
    )

    # Create per-class predictions and a macro-F1 report.
    print("\nGenerating per-class classification report...")

    validation_paths = []
    true_labels = []

    for label in ACTIVITIES:
        class_dir = DATASET_DIR / "val" / label

        for path in sorted(class_dir.iterdir()):
            if path.is_file():
                validation_paths.append(path)
                true_labels.append(label)

    predictions = best_model.predict(
        source=[str(path) for path in validation_paths],
        imgsz=IMAGE_SIZE,
        device="cpu",
        batch=BATCH_SIZE,
        verbose=False,
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

    pd.DataFrame(
        matrix,
        index=ACTIVITIES,
        columns=ACTIVITIES,
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    (OUTPUT_DIR / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    report_path = OUTPUT_DIR / "final_results.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write(
            "YOLOv12 Classification V2 - 3MDAD Day-RGB1\n"
        )
        report.write("=" * 60 + "\n")
        report.write(f"Architecture: {MODEL_CONFIG}\n")
        report.write("Initialization: From scratch\n")
        report.write(f"Best checkpoint: {best_checkpoint}\n")
        report.write(f"Training images: {sum(train_counts.values())}\n")
        report.write(
            f"Validation images: {sum(val_counts.values())}\n"
        )
        report.write(f"Image size: {IMAGE_SIZE}\n")
        report.write(f"Maximum epochs: {EPOCHS}\n")
        report.write(
            f"Elapsed minutes: "
            f"{(time.time() - start_time) / 60:.2f}\n"
        )
        report.write("\nUltralytics validation metrics:\n")

        for key, value in metrics.results_dict.items():
            report.write(f"{key}: {value}\n")

        report.write("\nPer-class classification report:\n")
        report.write(report_text)

    print("\n" + "=" * 68)
    print("YOLOv12 V2 EXPERIMENT COMPLETED")
    print("=" * 68)
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Training run: {run_dir}")
    print(f"Results: {report_path}")
    print(f"Classification report: {OUTPUT_DIR / 'classification_report.txt'}")
    print(f"Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.csv'}")
    print(
        f"Total elapsed time: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
