
from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\AI_FINAL_PROJECT\3MDAD")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

if not ROOT.exists():
    raise SystemExit(f"Dataset folder not found: {ROOT}")

print("=" * 75)
print("3MDAD ACTIVITY AND IMAGE COUNT REPORT")
print("=" * 75)

grand_total = 0

for stream in sorted(ROOT.iterdir()):
    if not stream.is_dir() or stream.name == "Annotations 3MDAD":
        continue

    counts = Counter()
    total = 0

    for image_path in stream.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Expected layout: stream / modality / subject / ACx / image
        activity = image_path.parent.name

        if activity.startswith("AC") and activity[2:].isdigit():
            counts[activity] += 1
            total += 1

    print(f"\n{stream.name}")
    print("-" * 45)

    for activity in sorted(
        counts,
        key=lambda name: int(name[2:])
    ):
        print(f"{activity:6} {counts[activity]:>10,} images")

    print(f"TOTAL  {total:>10,} images")
    grand_total += total

print("\n" + "=" * 75)
print(f"TOTAL IMAGES IN ACTIVITY FOLDERS: {grand_total:,}")
print("=" * 75)
