
from pathlib import Path
import pandas as pd
import random
import math

ROOT = Path(r"D:\AI_FINAL_PROJECT")
INPUT_CSV = ROOT / "3mdad_master_metadata.csv"
OUTPUT_CSV = ROOT / "3mdad_master_metadata_split.csv"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

if not INPUT_CSV.exists():
    raise SystemExit(
        f"Master metadata not found: {INPUT_CSV}\n"
        "Run prepare_3mdad_metadata.py first."
    )

if not math.isclose(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0
):
    raise SystemExit("Split ratios must sum to 1.0.")

df = pd.read_csv(INPUT_CSV)

required = {"image_path", "subject", "activity_id", "stream"}
missing = required - set(df.columns)

if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")

if df["subject"].isna().any():
    raise SystemExit("Subject IDs contain missing values.")

# Keep every subject entirely within one split.
subjects = sorted(df["subject"].astype(str).unique().tolist())

rng = random.Random(SEED)
rng.shuffle(subjects)

total_subjects = len(subjects)
train_end = round(total_subjects * TRAIN_RATIO)
val_end = train_end + round(total_subjects * VAL_RATIO)

# Make sure all subjects are assigned exactly once.
train_subjects = set(subjects[:train_end])
val_subjects = set(subjects[train_end:val_end])
test_subjects = set(subjects[val_end:])

if train_subjects & val_subjects:
    raise SystemExit("Subject leakage between train and validation.")

if train_subjects & test_subjects:
    raise SystemExit("Subject leakage between train and test.")

if val_subjects & test_subjects:
    raise SystemExit("Subject leakage between validation and test.")

def assign_split(subject):
    subject = str(subject)

    if subject in train_subjects:
        return "train"
    if subject in val_subjects:
        return "validation"
    return "test"

df["split"] = df["subject"].map(assign_split)

df.to_csv(OUTPUT_CSV, index=False)

print("=" * 60)
print("3MDAD SUBJECT-WISE SPLIT REPORT")
print("=" * 60)

print("Total image records:", f"{len(df):,}")
print("Unique subjects:", total_subjects)
print("Random seed:", SEED)

print("\nSubjects assigned:")
print("Train:", len(train_subjects))
print("Validation:", len(val_subjects))
print("Test:", len(test_subjects))

print("\nImage records per split:")
print(df["split"].value_counts().to_string())

print("\nActivity classes per split:")
for split_name in ["train", "validation", "test"]:
    subset = df[df["split"] == split_name]
    print(f"\n{split_name.upper()}")
    print(
        subset.groupby("activity_id")
        .size()
        .sort_index()
        .to_string()
    )

print("\nStreams per split:")
print(
    pd.crosstab(df["split"], df["stream"]).to_string()
)

print("\nChecking subject leakage...")
subject_sets = {
    name: set(df.loc[df["split"] == name, "subject"].astype(str))
    for name in ["train", "validation", "test"]
}

print(
    "Train/validation overlap:",
    len(subject_sets["train"] & subject_sets["validation"])
)
print(
    "Train/test overlap:",
    len(subject_sets["train"] & subject_sets["test"])
)
print(
    "Validation/test overlap:",
    len(subject_sets["validation"] & subject_sets["test"])
)

print("\nOutput:", OUTPUT_CSV)
print("Split generation complete.")
