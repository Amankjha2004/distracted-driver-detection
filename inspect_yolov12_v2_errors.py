
from pathlib import Path
import csv
import shutil
from ultralytics import YOLO

PROJECT = Path(r"D:\AI_FINAL_PROJECT")
BASE = PROJECT / "outputs" / "yolov12_cnn_v2"
DATASET = BASE / "dataset_v2"
MODEL_PATH = BASE / "runs" / "yolov12n_classification_v2" / "weights" / "best.pt"
OUT = PROJECT / "outputs" / "yolov12_v2_error_inspection"
COPIES = OUT / "misclassified_images"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"V2 checkpoint missing: {MODEL_PATH}")
    if not (DATASET / "val").is_dir():
        raise FileNotFoundError(f"Validation dataset missing: {DATASET / 'val'}")

    paths, actual = [], []
    for folder in sorted((DATASET / "val").iterdir()):
        if not folder.is_dir():
            continue
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in EXTENSIONS:
                paths.append(image)
                actual.append(folder.name)

    if not paths:
        raise RuntimeError("No validation images found.")

    OUT.mkdir(parents=True, exist_ok=True)
    COPIES.mkdir(parents=True, exist_ok=True)

    # Clear only the previous inspection copies, not the dataset or model.
    for item in COPIES.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=[str(p) for p in paths],
        imgsz=224,
        batch=8,
        device="cpu",
        verbose=False,
        stream=True,
    )

    errors = []
    for path, true_label, result in zip(paths, actual, results):
        index = int(result.probs.top1)
        predicted = result.names[index]
        confidence = float(result.probs.top1conf.item())

        if predicted != true_label:
            errors.append({
                "image_path": str(path),
                "actual_label": true_label,
                "predicted_label": predicted,
                "confidence": round(confidence, 5),
            })

    # Put errors predicted as safe driving first.
    errors.sort(key=lambda row: (
        row["predicted_label"] != "safe_driving",
        row["actual_label"],
        -row["confidence"],
    ))

    csv_path = OUT / "misclassified_images.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path", "actual_label",
                "predicted_label", "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(errors)

    # Copy up to 100 mistakes for visual review.
    for number, row in enumerate(errors[:100], start=1):
        source = Path(row["image_path"])
        name = (
            f"{number:03d}__actual-{row['actual_label']}"
            f"__pred-{row['predicted_label']}{source.suffix.lower()}"
        )
        shutil.copy2(source, COPIES / name)

    safe_errors = sum(
        row["predicted_label"] == "safe_driving" for row in errors
    )

    summary = (
        f"Validation images: {len(paths)}\n"
        f"Misclassified images: {len(errors)}\n"
        f"Incorrect predictions as safe_driving: {safe_errors}\n"
        f"CSV report: {csv_path}\n"
        f"Inspection images: {COPIES}\n"
    )
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
