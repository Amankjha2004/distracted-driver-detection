
from pathlib import Path
from collections import Counter
import csv
import re

ROOT = Path(r"D:\AI_FINAL_PROJECT\3MDAD")
OUTPUT = Path(r"D:\AI_FINAL_PROJECT\3mdad_master_metadata.csv")

ACTIVITIES = {
    1: "safe_driving",
    2: "hair_makeup",
    3: "adjusting_radio",
    4: "gps_operating",
    5: "texting_right_hand",
    6: "texting_left_hand",
    7: "phone_right_hand",
    8: "phone_left_hand",
    9: "taking_picture",
    10: "talking_to_passenger",
    11: "singing_or_dancing",
    12: "fatigue_somnolence",
    13: "drinking_right_hand",
    14: "drinking_left_hand",
    15: "reaching_behind",
    16: "smoking",
}

STREAMS = {
    "Day-RGB1": ("day", "RGB", "RGB1"),
    "Day-RGB2": ("day", "RGB", "RGB2"),
    "Day-Depth1": ("day", "Depth", "Depth1"),
    "Day-Depth2": ("day", "Depth", "Depth2"),
    "Night-IR1": ("night", "IR", "IR1"),
    "Night-IR2": ("night", "IR", "IR2"),
    "Night-Depth1": ("night", "Depth", "Depth1"),
    "Night-Depth2": ("night", "Depth", "Depth2"),
}

IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"
}

if not ROOT.exists():
    raise SystemExit(f"Dataset folder not found: {ROOT}")

fields = [
    "image_path",
    "stream",
    "time",
    "modality",
    "view",
    "subject",
    "activity_id",
    "activity_label",
    "filename",
]

total_images = 0
skipped_images = 0
stream_counts = Counter()
activity_counts = Counter()
subjects = set()

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for stream_name, settings in STREAMS.items():
        time_of_day, modality, view = settings
        stream_dir = ROOT / stream_name

        if not stream_dir.is_dir():
            print(f"WARNING: Missing stream: {stream_dir}")
            continue

        current_count = 0

        for image_path in stream_dir.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            activity_match = re.fullmatch(
                r"AC(\d+)", image_path.parent.name
            )

            if not activity_match:
                skipped_images += 1
                continue

            activity_id = int(activity_match.group(1))

            if activity_id not in ACTIVITIES:
                skipped_images += 1
                continue

            # Expected layout: stream/modality/subject/activity/image
            subject = image_path.parent.parent.name

            writer.writerow({
                "image_path": str(image_path.resolve()),
                "stream": stream_name,
                "time": time_of_day,
                "modality": modality,
                "view": view,
                "subject": subject,
                "activity_id": activity_id,
                "activity_label": ACTIVITIES[activity_id],
                "filename": image_path.name,
            })

            current_count += 1
            subjects.add((stream_name, subject))
            activity_counts[(stream_name, activity_id)] += 1

        stream_counts[stream_name] = current_count
        total_images += current_count

        print(f"{stream_name}: {current_count:,} images")

print("\n" + "=" * 60)
print("MASTER METADATA REPORT")
print("=" * 60)
print("Total image records:", f"{total_images:,}")
print("Images skipped:", f"{skipped_images:,}")
print("Streams found:", len(stream_counts))
print("Distinct stream-subject combinations:", len(subjects))
print("Output CSV:", OUTPUT)

print("\nImages per activity across each stream:")
for stream_name in STREAMS:
    if stream_name not in stream_counts:
        continue

    print(f"\n{stream_name}")
    for activity_id, label in ACTIVITIES.items():
        count = activity_counts[(stream_name, activity_id)]
        print(f"  AC{activity_id:02d} {label}: {count:,}")

print("\nMetadata generation complete.")
