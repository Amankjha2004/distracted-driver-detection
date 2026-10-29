from pathlib import Path
import numpy as np

root = Path(
    r"ThermalImagesDataset\ThermalImagesDataset-main"
    r"\Sources\lstm\data\sequences"
)

for class_name in ["normal", "yawning"]:
    files = sorted((root / class_name).glob("*.txt"))
    print(f"\n--- {class_name.upper()} ---")
    print("Files available:", len(files))

    for path in files[:3]:
        data = np.loadtxt(path, delimiter=";")
        print(
            path.name,
            "| shape:", data.shape,
            "| last-column values:", np.unique(data[:, -1]).tolist()
        )
