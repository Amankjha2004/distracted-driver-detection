
from pathlib import Path

ROOT = Path(r"D:\AI_FINAL_PROJECT\3MDAD")

print("=" * 60)
print("3MDAD DATASET STRUCTURE")
print("=" * 60)

if not ROOT.exists():
    print("Dataset folder not found:", ROOT)
    raise SystemExit(1)

for folder in sorted(ROOT.iterdir()):
    if folder.is_dir():
        print(f"\n[{folder.name}]")

        children = sorted(folder.iterdir())

        for child in children[:12]:
            if child.is_dir():
                print("  Folder:", child.name)

                subfolders = sorted(
                    item.name
                    for item in child.iterdir()
                    if item.is_dir()
                )

                if subfolders:
                    print("    Subfolders:", ", ".join(subfolders[:12]))

            else:
                print("  File:", child.name)

        if len(children) > 12:
            print(f"  ... and {len(children) - 12} more items")

print("\n" + "=" * 60)
print("SAMPLE IMAGE PATHS")
print("=" * 60)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
count = 0

for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in image_extensions:
        print(path.relative_to(ROOT))
        count += 1

        if count >= 25:
            break

if count == 0:
    print("No image files found.")
