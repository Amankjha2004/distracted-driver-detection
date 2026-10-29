
import sys
import torch
import torchvision
from pathlib import Path
from PIL import Image
import pandas as pd

print("=" * 60)
print("TRAINING ENVIRONMENT CHECK")
print("=" * 60)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU for training.")

csv_path = Path(
    r"D:\AI_FINAL_PROJECT\3mdad_master_metadata_split.csv"
)

if not csv_path.exists():
    raise SystemExit(f"Split CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

sample = df[
    (df["stream"] == "Day-RGB1")
    & (df["split"] == "train")
].head(1)

if sample.empty:
    raise SystemExit("No Day-RGB1 training sample found.")

image_path = Path(sample.iloc[0]["image_path"])

print("\nSample image:", image_path)
print("File exists:", image_path.is_file())

if image_path.is_file():
    with Image.open(image_path) as image:
        image.verify()

    print("Image verification: PASS")

# Test whether the pretrained ResNet50 architecture is available.
from torchvision.models import resnet50

model = resnet50(weights=None)
print("\nResNet50 model created successfully.")
print("Output classes before customization:", model.fc.out_features)

print("\nEnvironment check complete.")
