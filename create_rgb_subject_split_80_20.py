
import random
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
INPUT_CSV = "3mdad_master_metadata_split.csv"
OUTPUT_CSV = "3mdad_day_rgb1_subject_split_80_20.csv"

RANDOM_SEED = 42
TEST_SUBJECTS = 10
VALIDATION_SUBJECTS = 8

# --------------------------------------------------
# LOAD RGB DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df = df[df["stream"] == "Day-RGB1"].copy()

required = {"subject", "activity_label", "image_path", "stream"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[["subject", "activity_label", "image_path"]].isna().any().any():
    raise ValueError("Missing subject, label, or image_path values.")

subjects = sorted(df["subject"].unique().tolist())
classes = sorted(df["activity_label"].unique().tolist())

if len(subjects) != 50:
    raise ValueError(f"Expected 50 subjects, found {len(subjects)}.")

if len(classes) != 16:
    raise ValueError(f"Expected 16 classes, found {len(classes)}.")

# --------------------------------------------------
# SUBJECT-WISE SPLIT
# 40 subjects: training side (train + validation)
# 10 subjects: held-out test
# --------------------------------------------------
rng = random.Random(RANDOM_SEED)
shuffled = subjects.copy()
rng.shuffle(shuffled)

test_subjects = set(shuffled[:TEST_SUBJECTS])
remaining_subjects = shuffled[TEST_SUBJECTS:]

# Validation is taken only from the 40 training-side subjects.
validation_subjects = set(remaining_subjects[:VALIDATION_SUBJECTS])
train_subjects = set(remaining_subjects[VALIDATION_SUBJECTS:])

def assign_split(subject):
    if subject in test_subjects:
        return "test"
    if subject in validation_subjects:
        return "validation"
    return "train"

df["new_split"] = df["subject"].map(assign_split)

# --------------------------------------------------
# VALIDATION CHECKS
# --------------------------------------------------
subject_sets = {
    "train": train_subjects,
    "validation": validation_subjects,
    "test": test_subjects,
}

for name, group in df.groupby("new_split"):
    present_classes = set(group["activity_label"].unique())
    missing_classes = set(classes) - present_classes

    print(f"\n{name.upper()}")
    print("Subjects:", group["subject"].nunique())
    print("Images:", len(group))
    print("Classes present:", len(present_classes))

    if missing_classes:
        raise ValueError(
            f"{name} is missing classes: {sorted(missing_classes)}. "
            "Do not train yet; adjust the subject selection."
        )

# Confirm no subject appears in multiple splits.
for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    overlap = subject_sets[a] & subject_sets[b]
    if overlap:
        raise ValueError(f"Subject leakage between {a} and {b}: {overlap}")

# Save to a NEW file; the original CSV remains untouched.
df.to_csv(OUTPUT_CSV, index=False)

print("\nSUBJECT COUNTS")
for name, group in df.groupby("new_split"):
    print(f"{name}: {group['subject'].nunique()} subjects")

print("\nCLASS COUNTS BY SPLIT")
print(pd.crosstab(df["new_split"], df["activity_label"]))

print(f"\nSaved new metadata file: {Path(OUTPUT_CSV).resolve()}")
print("Original CSV was not modified.")
