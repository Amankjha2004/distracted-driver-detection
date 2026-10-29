"""
Create a master comparison table for the five completed 3MDAD Day-RGB1 experiments.
This only consolidates recorded metrics; it does not train models or use test data.

Run:
    cd D:\AI_FINAL_PROJECT
    .\.venv\Scripts\python.exe create_master_results.py
"""
import csv
from pathlib import Path

ROOT = Path(r"D:\AI_FINAL_PROJECT")
OUT_DIR = ROOT / "outputs" / "master_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics transcribed from the experiment outputs shared so far.
rows = [
    {
        "algorithm": "YOLOv12 + CNN classifier (V2)",
        "dataset": "3MDAD Day-RGB1",
        "train_images": 4480,
        "validation_images": 1024,
        "validation_accuracy": 0.5537109375,
        "macro_f1": 0.5389,
        "weighted_f1": "",
        "runtime_minutes": 95.72,
        "checkpoint": r"outputs\yolov12_cnn_v2\runs\yolov12n_classification_v2\weights\best.pt",
        "held_out_test_used": "No",
        "comparison_note": "V2 result; 30 maximum epochs",
    },
    {
        "algorithm": "ResNet50 + BiLSTM",
        "dataset": "3MDAD Day-RGB1",
        "train_images": 400,
        "validation_images": 128,
        "validation_accuracy": 0.4141,
        "macro_f1": 0.4129,
        "weighted_f1": "",
        "runtime_minutes": "",
        "checkpoint": r"outputs\resnet50_bilstm_improved\best_resnet50_bilstm.pth",
        "held_out_test_used": "No",
        "comparison_note": "Sequence-based experiment; validation setup differs",
    },
    {
        "algorithm": "DenseNet121 + SVM (RBF, C=1)",
        "dataset": "3MDAD Day-RGB1",
        "train_images": 4480,
        "validation_images": 1024,
        "validation_accuracy": 0.3779296875,
        "macro_f1": 0.3851553085746131,
        "weighted_f1": 0.3851553085746131,
        "runtime_minutes": 20.47,
        "checkpoint": r"outputs\densenet121_svm\best_densenet121_svm.joblib",
        "held_out_test_used": "No",
        "comparison_note": "RBF SVM selected using validation macro-F1",
    },
    {
        "algorithm": "EfficientNetB3 + ViT",
        "dataset": "3MDAD Day-RGB1",
        "train_images": 1600,
        "validation_images": 320,
        "validation_accuracy": 0.3750,
        "macro_f1": 0.3334,
        "weighted_f1": "",
        "runtime_minutes": "",
        "checkpoint": r"outputs\efficientnetb3_vit\best_efficientnetb3_vit.pth",
        "held_out_test_used": "No",
        "comparison_note": "Feature-extraction/fusion experiment; validation setup differs",
    },
    {
        "algorithm": "MobileNetV3-Large + Channel/Spatial Attention",
        "dataset": "3MDAD Day-RGB1",
        "train_images": 4480,
        "validation_images": 1024,
        "validation_accuracy": 0.33984375,
        "macro_f1": 0.3307468286425177,
        "weighted_f1": 0.3307468286425177,
        "runtime_minutes": 18.37,
        "checkpoint": r"outputs\mobilenetv3_attention\best_mobilenetv3_attention.pth",
        "held_out_test_used": "No",
        "comparison_note": "Best checkpoint selected by validation macro-F1",
    },
]

csv_path = OUT_DIR / "master_results.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

readme_path = OUT_DIR / "MASTER_RESULTS_README.txt"
lines = [
    "MASTER RESULTS — DISTRACTED DRIVER DETECTION",
    "Dataset: 3MDAD Day-RGB1",
    "",
    "IMPORTANT:",
    "- These are validation results, not held-out test results.",
    "- Validation sizes and experiment setups differ.",
    "- Do not claim a definitive fair ranking until all models are evaluated",
    "  on the same fixed validation examples with consistent preprocessing.",
    "- Blank values mean the metric/runtime was not available in the recorded results.",
    "- The held-out test set was not used for these experiments.",
    "",
    "Recorded metrics (preliminary ranking by recorded accuracy):",
]
for row in sorted(rows, key=lambda r: r["validation_accuracy"], reverse=True):
    lines.append(
        f"- {row['algorithm']}: accuracy={row['validation_accuracy']:.2%}, "
        f"macro-F1={row['macro_f1']:.4f}, validation_images={row['validation_images']}"
    )
lines.extend(["", f"CSV file: {csv_path}"])
readme_path.write_text("\n".join(lines), encoding="utf-8")

print("MASTER RESULTS TABLE CREATED")
print(f"CSV: {csv_path}")
print(f"Notes: {readme_path}")
print("\nPreliminary ranking by recorded validation accuracy:")
for row in sorted(rows, key=lambda r: r["validation_accuracy"], reverse=True):
    print(
        f"{row['algorithm']}: accuracy={row['validation_accuracy']:.2%}, "
        f"macro-F1={row['macro_f1']:.4f}, val_images={row['validation_images']}"
    )
print("\nNext: standardize evaluation on the same validation images before final claims.")
