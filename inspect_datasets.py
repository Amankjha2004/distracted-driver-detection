
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "State Farm": BASE_DIR / "state-farm-distracted-driver-detection",
    "3MDAD": BASE_DIR / "3MDAD",
    "ThermalImagesDataset": (
        BASE_DIR / "ThermalImagesDataset" / "ThermalImagesDataset-main"
    ),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(folder):
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


for name, folder in DATASETS.items():
    print("\n" + "=" * 60)
    print(f"Dataset: {name}")
    print(f"Path: {folder}")

    if not folder.is_dir():
        print("STATUS: Folder not found. Check the path.")
        continue

    print("STATUS: Folder found")
    print(f"Total images: {count_images(folder):,}")

    if name == "State Farm":
        train_dir = folder / "imgs" / "train"

        if train_dir.is_dir():
            print("\nTraining classes:")
            for class_dir in sorted(train_dir.iterdir()):
                if class_dir.is_dir():
                    print(
                        f"  {class_dir.name}: "
                        f"{count_images(class_dir):,} images"
                    )
        else:
            print("Training folder not found at imgs/train.")

    elif name == "3MDAD":
        print("\nTop-level folders:")
        for item in sorted(folder.iterdir()):
            if item.is_dir():
                print(f"  {item.name}")

print("\nDataset inspection complete.")
