
from pathlib import Path
import pandas as pd
from scipy.io import loadmat

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATE_FARM_DIR = (
    BASE_DIR / "state-farm-distracted-driver-detection"
)

MDAD_DIR = BASE_DIR / "3MDAD"

THERMAL_DIR = (
    BASE_DIR / "ThermalImagesDataset"
    / "ThermalImagesDataset-main"
)


# ============================================================
# 1. STATE FARM: LABEL INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("STATE FARM: DRIVER LABELS")

csv_path = STATE_FARM_DIR / "driver_imgs_list.csv"

if csv_path.is_file():
    try:
        df = pd.read_csv(csv_path)

        print("CSV path:", csv_path)
        print("Columns:", list(df.columns))

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        if "classname" in df.columns:
            print("\nImages per class:")
            print(
                df["classname"]
                .value_counts()
                .sort_index()
                .to_string()
            )

        if "subject" in df.columns:
            print(
                "\nNumber of unique drivers:",
                df["subject"].nunique()
            )

    except Exception as error:
        print("Could not read State Farm CSV:", error)
else:
    print("CSV not found:", csv_path)


# ============================================================
# 2. 3MDAD: MATLAB ANNOTATIONS
# ============================================================

print("\n" + "=" * 60)
print("3MDAD: MATLAB ANNOTATIONS")

annotation_dir = MDAD_DIR / "Annotations 3MDAD"

if annotation_dir.is_dir():
    print("Annotation directory:", annotation_dir)

    # Search recursively in case the MAT files are nested.
    mat_files = sorted(annotation_dir.rglob("*.mat"))

    # Also handle uppercase extensions if present.
    if not mat_files:
        mat_files = sorted(annotation_dir.rglob("*.MAT"))

    print("MAT files found:", len(mat_files))

    if not mat_files:
        print("No MATLAB files found. Checking folder contents:")

        for item in list(annotation_dir.iterdir())[:20]:
            print(" ", item.name)

    # Inspect only the first five files.
    for path in mat_files[:5]:
        print("\n" + "-" * 50)
        print("File:", path.relative_to(annotation_dir))

        try:
            data = loadmat(path)

            keys = [
                key for key in data
                if not key.startswith("__")
            ]

            print("Variables:", keys)

            for key in keys:
                value = data[key]

                print(
                    f"  Variable: {key} | "
                    f"Shape: {getattr(value, 'shape', 'unknown')} | "
                    f"Type: {type(value).__name__}"
                )

                print("  Sample:", repr(value)[:300])

        except Exception as error:
            print("Could not read file:", error)

else:
    print("Annotation directory not found:", annotation_dir)

    if MDAD_DIR.is_dir():
        print("Available folders inside 3MDAD:")
        for item in MDAD_DIR.iterdir():
            print(" ", item.name)


# ============================================================
# 3. THERMAL DATASET: ANNOTATIONS
# ============================================================

print("\n" + "=" * 60)
print("THERMAL DATASET: ANNOTATIONS")

if THERMAL_DIR.is_dir():
    annotation_files = sorted(
        THERMAL_DIR.rglob("*annotations.txt")
    )

    print("Thermal directory:", THERMAL_DIR)
    print("Annotation files found:", len(annotation_files))

    for path in annotation_files[:5]:
        print("\n" + "-" * 50)
        print("File:", path.relative_to(THERMAL_DIR))

        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace"
            ) as file:
                for line_number, line in enumerate(file, start=1):
                    if line_number > 5:
                        break

                    print(line.rstrip()[:250])

        except OSError as error:
            print("Could not read annotation file:", error)

else:
    print("Thermal dataset folder not found:", THERMAL_DIR)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("Label inspection complete.")
