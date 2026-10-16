
from pathlib import Path
import pandas as pd

CSV_PATH = Path(r"D:\AI_FINAL_PROJECT\3mdad_master_metadata.csv")

if not CSV_PATH.exists():
    raise SystemExit(
        f"Master CSV not found: {CSV_PATH}\n"
        "Run prepare_3mdad_metadata.py first."
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image_path",
    "stream",
    "time",
    "modality",
    "view",
    "subject",
    "activity_id",
    "activity_label",
    "filename",
}

missing = required_columns - set(df.columns)

if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

print("=" * 60)
print("3MDAD MASTER CSV VALIDATION")
print("=" * 60)

print("Total records:", f"{len(df):,}")
print("Unique image paths:", f"{df['image_path'].nunique():,}")
print("Streams:", df["stream"].nunique())
print("Activity classes:", df["activity_id"].nunique())
print("Subjects across streams:", df.groupby("stream")["subject"].nunique().to_dict())

print("\nRecords per stream:")
print(df["stream"].value_counts().sort_index().to_string())

print("\nRecords per activity:")
print(
    df.groupby(["activity_id", "activity_label"])
      .size()
      .to_string()
)

print("\nMissing values per column:")
print(df[list(required_columns)].isna().sum().to_string())

print("\nChecking whether sample image paths exist...")
sample = df["image_path"].dropna().head(100)
missing_paths = [path for path in sample if not Path(path).is_file()]

print("Paths checked:", len(sample))
print("Missing paths in sample:", len(missing_paths))

if missing_paths:
    print("Examples:")
    for path in missing_paths[:5]:
        print(path)

print("\nValidation complete.")
