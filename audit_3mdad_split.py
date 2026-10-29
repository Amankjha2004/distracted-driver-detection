
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\AI_FINAL_PROJECT")
CSV_PATH = ROOT / "3mdad_master_metadata_split.csv"

EXPECTED_CLASSES = set(range(1, 17))
EXPECTED_STREAMS = {
    "Day-Depth1",
    "Day-Depth2",
    "Day-RGB1",
    "Day-RGB2",
    "Night-Depth1",
    "Night-Depth2",
    "Night-IR1",
    "Night-IR2",
}
SPLITS = ["train", "validation", "test"]

if not CSV_PATH.exists():
    raise SystemExit(f"Split CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required = {
    "image_path", "stream", "subject",
    "activity_id", "activity_label", "split"
}

missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

print("=" * 70)
print("3MDAD SPLIT AUDIT")
print("=" * 70)

print("\n1. BASIC CHECKS")
print("Total records:", f"{len(df):,}")
print("Duplicate image paths:", df["image_path"].duplicated().sum())
print("Missing values in required columns:")
print(df[list(required)].isna().sum().to_string())

actual_streams = set(df["stream"].unique())
print("\nExpected streams:", len(EXPECTED_STREAMS))
print("Found streams:", len(actual_streams))

if actual_streams != EXPECTED_STREAMS:
    print("Missing streams:", sorted(EXPECTED_STREAMS - actual_streams))
    print("Unexpected streams:", sorted(actual_streams - EXPECTED_STREAMS))

print("\n2. SUBJECT COUNTS BY STREAM AND SPLIT")
subject_counts = (
    df.groupby(["stream", "split"])["subject"]
      .nunique()
      .unstack(fill_value=0)
      .reindex(columns=SPLITS, fill_value=0)
)
print(subject_counts.to_string())

print("\n3. IMAGE COUNTS BY STREAM AND SPLIT")
image_counts = (
    df.groupby(["stream", "split"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=SPLITS, fill_value=0)
)
print(image_counts.to_string())

print("\n4. ACTIVITY COVERAGE BY STREAM AND SPLIT")
for stream in sorted(actual_streams):
    print(f"\n{stream}")

    subset = df[df["stream"] == stream]

    for split in SPLITS:
        part = subset[subset["split"] == split]
        classes = set(part["activity_id"].dropna().astype(int))
        missing_classes = EXPECTED_CLASSES - classes

        print(
            f"  {split:10} "
            f"classes={len(classes):2}/16, "
            f"images={len(part):,}, "
            f"missing={sorted(missing_classes)}"
        )

print("\n5. SUBJECT OVERLAP BETWEEN SPLITS")
subject_sets = {
    split: set(
        df.loc[df["split"] == split, "subject"]
          .dropna().astype(str)
    )
    for split in SPLITS
}

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

for first, second in pairs:
    overlap = subject_sets[first] & subject_sets[second]
    print(f"{first} vs {second}: {len(overlap)} overlapping IDs")

print("\n6. SUBJECT ID FORMAT")
for stream in sorted(actual_streams):
    subjects = sorted(
        df.loc[df["stream"] == stream, "subject"]
          .dropna().astype(str).unique()
    )
    print(
        f"{stream}: {len(subjects)} subjects; "
        f"examples={subjects[:8]}"
    )

print("\n7. IMAGE PATH CHECK")
sample_paths = df["image_path"].dropna().head(200)
missing_paths = [
    path for path in sample_paths
    if not Path(path).is_file()
]

print("Paths checked:", len(sample_paths))
print("Missing paths in sample:", len(missing_paths))

if missing_paths:
    for path in missing_paths[:5]:
        print("Missing:", path)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("No files were modified.")
print("=" * 70)
