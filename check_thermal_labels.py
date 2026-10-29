
from pathlib import Path
import re
import numpy as np

ROOT = Path(
    r"D:\AI_FINAL_PROJECT\ThermalImagesDataset"
    r"\ThermalImagesDataset-main\Sources\lstm\data\sequences"
)

folders = ["normal", "yawning"]
checked = 0
mismatches = []

for folder in folders:
    expected_folder_label = 0 if folder == "normal" else 1

    for path in sorted((ROOT / folder).glob("*.txt")):
        # Example:
        # 1_Man_Glasses1_Yawning_face_0_188-219.txt
        match = re.match(r"(.+)_face_\d+_(\d+)-(\d+)$", path.stem)

        if not match:
            continue

        recording = match.group(1)
        start_frame = int(match.group(2))
        end_frame = int(match.group(3))

        try:
            data = np.loadtxt(path, delimiter=";", dtype=float)
        except Exception:
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.ndim != 2 or data.shape[1] != 4:
            continue

        labels = data[:, 3]
        if not np.isin(labels, [0, 1]).all():
            continue

        # Inspect only the recording name and the prepared labels.
        # This does not assume the folder name is the ground truth.
        positive = int(np.sum(labels == 1))
        negative = int(np.sum(labels == 0))

        if folder == "normal" and positive > 0:
            mismatches.append(
                (path.name, recording, start_frame, end_frame,
                 negative, positive)
            )

        checked += 1

print("Sequences checked:", checked)
print("Normal-folder sequences containing positive frame labels:",
      len(mismatches))

for item in mismatches[:20]:
    print(item)
