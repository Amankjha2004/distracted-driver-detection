
import os
import re
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\AI_FINAL_PROJECT"

CSV_PATH = os.path.join(
    PROJECT_DIR,
    "3mdad_master_metadata_split.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "sequence_inspection",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

STREAM_NAME = "Day-RGB1"

# ============================================================
# FRAME NUMBER EXTRACTION
# ============================================================

def get_frame_number(image_path):
    filename = Path(str(image_path)).stem

    matches = re.findall(
        r"F(\d+)",
        filename,
        flags=re.IGNORECASE,
    )

    return int(matches[-1]) if matches else -1


# ============================================================
# LOAD METADATA
# ============================================================

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"Metadata CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image_path",
    "stream",
    "split",
    "subject",
    "activity_id",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing CSV columns: {missing}"
    )

df["stream"] = df["stream"].astype(str).str.strip()
df["split"] = (
    df["split"].astype(str).str.strip().str.lower()
)
df["subject"] = df["subject"].astype(str).str.strip()
df["activity_id"] = pd.to_numeric(
    df["activity_id"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "image_path",
        "stream",
        "split",
        "subject",
        "activity_id",
    ]
).copy()

df["activity_id"] = df["activity_id"].astype(int)

df["split"] = df["split"].replace({
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "testing": "test",
})

df = df[
    (df["stream"] == STREAM_NAME)
    & df["activity_id"].between(1, 16)
].copy()

if df.empty:
    raise ValueError(
        f"No data found for stream {STREAM_NAME}"
    )

# The complete directory path is important:
# each subject/activity directory should be distinct.
df["recording_key"] = df["image_path"].map(
    lambda p: str(Path(str(p)).parent)
)

df["frame_number"] = df["image_path"].map(
    get_frame_number
)

df["file_exists"] = df["image_path"].map(
    os.path.isfile
)

# ============================================================
# BASIC DATA SUMMARY
# ============================================================

print("=" * 65)
print("3MDAD TEMPORAL SEQUENCE INSPECTION")
print("=" * 65)

print("\nStream:", STREAM_NAME)
print("Total metadata records:", len(df))
print("Existing image files:", int(df["file_exists"].sum()))
print("Missing image files:", int((~df["file_exists"]).sum()))

print("\nRecords by split:")
print(df["split"].value_counts())

print("\nSubjects by split:")
print(
    df.groupby("split")["subject"]
    .nunique()
)

print("\nRecording folders by split:")
print(
    df.groupby("split")["recording_key"]
    .nunique()
)

print("\nActivity counts by split:")
print(
    pd.crosstab(
        df["split"],
        df["activity_id"],
    ).to_string()
)

# ============================================================
# RECORDING-LEVEL INSPECTION
# ============================================================

recording_rows = []

for recording_key, group in df.groupby(
    "recording_key",
    sort=True,
):
    group = group.sort_values(
        ["frame_number", "image_path"]
    )

    frame_numbers = group["frame_number"].to_numpy()
    valid_numbers = frame_numbers[frame_numbers >= 0]

    unique_numbers = sorted(set(valid_numbers.tolist()))

    if len(unique_numbers) >= 2:
        differences = pd.Series(
            unique_numbers
        ).diff().dropna()

        consecutive_pairs = int((differences == 1).sum())
        gaps = int((differences > 1).sum())
    else:
        consecutive_pairs = 0
        gaps = 0

    if len(group) > 0:
        first_path = str(group.iloc[0]["image_path"])
        last_path = str(group.iloc[-1]["image_path"])

        subject = str(group.iloc[0]["subject"])
        activity_id = int(group.iloc[0]["activity_id"])
        split = str(group.iloc[0]["split"])
    else:
        first_path = ""
        last_path = ""
        subject = ""
        activity_id = -1
        split = ""

    recording_rows.append({
        "recording_key": recording_key,
        "subject": subject,
        "activity_id": activity_id,
        "split": split,
        "image_count": len(group),
        "valid_frame_numbers": len(valid_numbers),
        "unique_frame_numbers": len(unique_numbers),
        "first_frame_number": (
            min(valid_numbers) if len(valid_numbers) else -1
        ),
        "last_frame_number": (
            max(valid_numbers) if len(valid_numbers) else -1
        ),
        "consecutive_frame_pairs": consecutive_pairs,
        "frame_number_gaps": gaps,
        "first_path": first_path,
        "last_path": last_path,
    })

recordings = pd.DataFrame(recording_rows)

recordings_path = os.path.join(
    OUTPUT_DIR,
    "recording_inspection.csv",
)

recordings.to_csv(
    recordings_path,
    index=False,
)

print("\nRecording-level summary:")
print(
    recordings.groupby("split")["image_count"]
    .agg(["count", "min", "median", "max"])
    .to_string()
)

print("\nRecordings with fewer than 8 images:")
print(
    int((recordings["image_count"] < 8).sum())
)

print("\nRecordings with fewer than 8 valid frame numbers:")
print(
    int((recordings["valid_frame_numbers"] < 8).sum())
)

print("\nRecordings with no consecutive frame pairs:")
print(
    int((recordings["consecutive_frame_pairs"] == 0).sum())
)

# ============================================================
# CHECK SUBJECT / ACTIVITY CONSISTENCY
# ============================================================

subject_counts = (
    df.groupby("recording_key")["subject"]
    .nunique()
)

activity_counts = (
    df.groupby("recording_key")["activity_id"]
    .nunique()
)

print("\nRecording consistency:")
print(
    "Folders containing multiple subjects:",
    int((subject_counts > 1).sum()),
)

print(
    "Folders containing multiple activity labels:",
    int((activity_counts > 1).sum()),
)

# ============================================================
# SHOW SAMPLE RECORDINGS
# ============================================================

print("\nSample recording folders:")

columns_to_show = [
    "subject",
    "activity_id",
    "split",
    "image_count",
    "valid_frame_numbers",
    "first_frame_number",
    "last_frame_number",
    "consecutive_frame_pairs",
]

print(
    recordings[columns_to_show]
    .head(20)
    .to_string(index=False)
)

# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

subject_sets = {
    split: set(group["subject"].unique())
    for split, group in df.groupby("split")
}

print("\nSubject overlap checks:")

split_names = sorted(subject_sets.keys())

for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        first = split_names[i]
        second = split_names[j]

        overlap = subject_sets[first] & subject_sets[second]

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} shared subjects"
        )

# ============================================================
# SAVE A COMPACT SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "inspection_summary.txt",
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("3MDAD Temporal Sequence Inspection\n")
    file.write("=" * 45 + "\n")
    file.write(f"Stream: {STREAM_NAME}\n")
    file.write(f"Metadata records: {len(df)}\n")
    file.write(
        f"Existing images: {int(df['file_exists'].sum())}\n"
    )
    file.write(
        f"Missing images: {int((~df['file_exists']).sum())}\n"
    )
    file.write(
        f"Recording folders: {len(recordings)}\n"
    )
    file.write(
        "Recordings with fewer than 8 images: "
        f"{int((recordings['image_count'] < 8).sum())}\n"
    )
    file.write(
        "Recordings with fewer than 8 valid frame numbers: "
        f"{int((recordings['valid_frame_numbers'] < 8).sum())}\n"
    )
    file.write(
        "Recordings with no consecutive frame pairs: "
        f"{int((recordings['consecutive_frame_pairs'] == 0).sum())}\n"
    )
    file.write("\nRecords by split:\n")
    file.write(df["split"].value_counts().to_string())
    file.write("\n\nRecording summary by split:\n")
    file.write(
        recordings.groupby("split")["image_count"]
        .agg(["count", "min", "median", "max"])
        .to_string()
    )

print("\nInspection completed.")
print("Recording report:", recordings_path)
print("Summary report:", summary_path)
