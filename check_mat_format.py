
from pathlib import Path
import h5py
from scipy.io import loadmat

MAT_DIR = Path(
    r"D:\AI_FINAL_PROJECT\3MDAD\Annotations 3MDAD"
)

files = sorted(MAT_DIR.rglob("*.mat"))

if not files:
    files = sorted(MAT_DIR.rglob("*.MAT"))

if not files:
    print("No MATLAB annotation files found.")
    raise SystemExit(1)

file_path = files[0]

print("Checking:", file_path)
print("File size:", file_path.stat().st_size, "bytes")

# Check for MATLAB v7.3 / HDF5 format
try:
    with h5py.File(file_path, "r") as f:
        print("\nFormat: HDF5 / likely MATLAB v7.3")
        print("Top-level keys:", list(f.keys()))

        def show_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(
                    "Dataset:",
                    name,
                    "| Shape:",
                    obj.shape,
                    "| Type:",
                    obj.dtype
                )
            elif isinstance(obj, h5py.Group):
                print("Group:", name)

        f.visititems(show_structure)

except OSError:
    print("\nNot readable as HDF5.")

    try:
        data = loadmat(file_path)
        print("Format: Traditional MATLAB MAT file")
        print("Variables:", list(data.keys()))

        for name, value in data.items():
            if not name.startswith("__"):
                print(
                    name,
                    "| Shape:", getattr(value, "shape", None),
                    "| Type:", type(value).__name__
                )

    except Exception as error:
        print("Could not decode file:", error)
