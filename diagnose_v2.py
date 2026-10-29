
from pathlib import Path
import random
from collections import defaultdict
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "outputs" / "yolov12_cnn_v2" / "runs"
    / "yolov12n_classification_v2" / "weights" / "best.pt"
)
VAL_DIR = ROOT / "outputs" / "yolov12_cnn_v2" / "dataset_v2" / "val"

random.seed(42)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in VAL_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

if not images:
    raise RuntimeError("No validation images found.")

sample = random.sample(images, min(100, len(images)))

model = YOLO(str(MODEL_PATH))
results = model.predict(
    source=[str(p) for p in sample],
    imgsz=224,
    batch=8,
    verbose=False
)

correct = 0
class_stats = defaultdict(lambda: [0, 0])

print(f"\nImages tested: {len(results)}\n")

for path, result in zip(sample, results):
    actual = path.parent.name
    predicted = model.names[int(result.probs.top1)]
    confidence = float(result.probs.top1conf) * 100

    is_correct = actual == predicted
    correct += is_correct
    class_stats[actual][1] += 1
    class_stats[actual][0] += int(is_correct)

    print(
        f"{'OK' if is_correct else 'WRONG':5} | "
        f"Actual: {actual:24} | "
        f"Predicted: {predicted:24} | "
        f"Confidence: {confidence:.1f}%"
    )

print("\n========== SUMMARY ==========")
print(f"Sample accuracy: {correct / len(results) * 100:.2f}%")
print("\nAccuracy by actual class:")

for name in sorted(class_stats):
    right, total = class_stats[name]
    print(f"{name:24}: {right}/{total} ({right / total * 100:.1f}%)")
