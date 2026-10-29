
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"D:\AI_FINAL_PROJECT")

DATASET = ROOT / "outputs" / "yolov12_cnn_v2" / "dataset_v2"

CHECKPOINTS = {
    "V2 best": (
        ROOT / "outputs" / "yolov12_cnn_v2" / "runs"
        / "yolov12n_classification_v2" / "weights" / "best.pt"
    ),
    "V2 last": (
        ROOT / "outputs" / "yolov12_cnn_v2" / "runs"
        / "yolov12n_classification_v2" / "weights" / "last.pt"
    ),
}

if not DATASET.is_dir():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

for name, checkpoint in CHECKPOINTS.items():
    print(f"\n{'=' * 50}")
    print(name)
    print("Checkpoint:", checkpoint)

    if not checkpoint.is_file():
        print("Checkpoint missing; skipping.")
        continue

    model = YOLO(str(checkpoint))

    metrics = model.val(
        data=str(DATASET),
        split="val",
        imgsz=224,
        batch=8,
        device="cpu",
        workers=0,
        plots=False,
    )

    print(f"Top-1 accuracy: {metrics.top1:.4f}")
    print(f"Top-5 accuracy: {metrics.top5:.4f}")
