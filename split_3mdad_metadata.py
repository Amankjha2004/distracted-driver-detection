
from pathlib import Path
import random
import pandas as pd

ROOT = Path(r"D:\AI_FINAL_PROJECT")
INPUT_CSV = ROOT / "3mdad_master_metadata.csv"
OUTPUT_CSV = ROOT / "3mdad_master_metadata_split.csv"

SEED = 42

SPLIT_NAMES = ["train", "validation", "test"]

def split_subjects(subjects, seed):
    """Split subjects reproducibly: 70% train, 15% validation, rest test."""
    subjects = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    train_end = round(n * 0.70)
    val_end = train_end + round(n * 0.15)

    return {
        "train": set(subjects[:train_end]),
        "validation": set(subjects[train_end:val_end]),
        "test": set(subjects[val_end:]),
    }

if not INPUT_CSV.exists():
    raise SystemExit(f"Metadata CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

required = {"image_path", "stream", "time", "subject", "activity_id"}
missing = required - set(df.columns)

if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

if df[list(required)].isna().any().any():
    raise SystemExit("Required metadata contains missing values.")

# Day and night were collected as separate participant cohorts.
# Keep each subject together across every modality/view within its cohort.
df["subject"] = df["subject"].astype(str)
df["time"] = df["time"].astype(str).str.lower()

if not set(df["time"].unique()).issubset({"day", "night"}):
    raise SystemExit("Unexpected time/cohort value in metadata.")

assignments = {}

for cohort_index, cohort in enumerate(["day", "night"]):
    cohort_df = df[df["time"] == cohort]
    subjects = sorted(cohort_df["subject"].unique())

    if not subjects:
        print(f"WARNING: No subjects found for {cohort}.")
        continue

    groups = split_subjects(subjects, SEED + cohort_index)

    print(f"\n{cohort.upper()} cohort: {len(subjects)} subjects")

    for split_name, split_subjects_set in groups.items():
        print(f"  {split_name}: {len(split_subjects_set)} subjects")

        for subject in split_subjects_set:
            assignments[(cohort, subject)] = split_name

df["split"] = [
    assignments[(time, subject)]
    for time, subject in zip(df["time"], df["subject"])
]

# Verify each cohort/subject has only one split.
subject_split_counts = (
    df.groupby(["time", "subject"])["split"].nunique()
)

if (subject_split_counts > 1).any():
    raise SystemExit("ERROR: A subject occurs in multiple splits.")

# Verify no image path is duplicated.
if df["image_path"].duplicated().any():
    raise SystemExit("ERROR: Duplicate image paths found.")

df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 65)
print("UPDATED 3MDAD COHORT-AWARE SPLIT REPORT")
print("=" * 65)
print("Total records:", f"{len(df):,}")
print("Output:", OUTPUT_CSV)

print("\nImage records per cohort and split:")
print(
    pd.crosstab(df["time"], df["split"])
      .reindex(columns=SPLIT_NAMES, fill_value=0)
      .to_string()
)

print("\nSubjects per stream and split:")
print(
    df.groupby(["stream", "split"])["subject"]
      .nunique()
      .unstack(fill_value=0)
      .reindex(columns=SPLIT_NAMES, fill_value=0)
      .to_string()
)

print("\nActivity coverage:")
for (cohort, split_name), part in df.groupby(["time", "split"]):
    classes = set(part["activity_id"].astype(int))
    print(
        f"{cohort:5} {split_name:10}: "
        f"{len(classes)}/16 classes"
    )

print("\nSubject overlap checks within each cohort:")
for cohort in ["day", "night"]:
    sets = {
        split_name: set(
            df.loc[
                (df["time"] == cohort) & (df["split"] == split_name),
                "subject"
            ]
        )
        for split_name in SPLIT_NAMES
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = sets[first] & sets[second]
        print(
            f"{cohort}: {first} vs {second}: "
            f"{len(overlap)} overlapping subjects"
        )

print("\nSplit generation complete.")
