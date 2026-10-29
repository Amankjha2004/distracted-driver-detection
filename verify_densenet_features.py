
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(r"D:\AI_FINAL_PROJECT")
VAL_DIR = ROOT / "outputs" / "yolov12_cnn_v2" / "dataset_v2" / "val"
CACHE_PATH = ROOT / "outputs" / "densenet121_svm" / "val_features.npz"

BATCH_SIZE = 8
DEVICE = torch.device("cpu")

def main():
    print("Loading validation images...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    dataset = datasets.ImageFolder(str(VAL_DIR), transform=transform)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    print("Validation images:", len(dataset))
    print("Class order:", dataset.classes)

    print("\nLoading pretrained DenseNet121...")
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )
    model.classifier = nn.Identity()
    model.eval().to(DEVICE)

    features_list = []
    labels_list = []

    print("\nExtracting features from actual images...")
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            features = model(images.to(DEVICE))
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

            if batch_idx % 16 == 0 or batch_idx == len(loader):
                print(
                    f"Processed "
                    f"{min(batch_idx * BATCH_SIZE, len(dataset))}"
                    f"/{len(dataset)} images"
                )

    fresh_features = np.concatenate(features_list, axis=0)
    fresh_labels = np.concatenate(labels_list, axis=0)

    print("\nLoading existing cache...")
    with np.load(CACHE_PATH, allow_pickle=False) as saved:
        cached_features = saved["features"]
        cached_labels = saved["labels"].astype(np.int64)

    print("\n=== VERIFICATION RESULTS ===")
    print("Fresh feature shape:", fresh_features.shape)
    print("Cached feature shape:", cached_features.shape)
    print("Fresh labels match cached labels:",
          np.array_equal(fresh_labels, cached_labels))

    if fresh_features.shape != cached_features.shape:
        print("Feature shapes differ; cache is incompatible.")
        return

    difference = np.abs(
        fresh_features.astype(np.float64)
        - cached_features.astype(np.float64)
    )

    print("Maximum absolute feature difference:", difference.max())
    print("Mean absolute feature difference:", difference.mean())
    print("Features match within tolerance:",
          np.allclose(
              fresh_features, cached_features,
              rtol=1e-4, atol=1e-5
          ))

if __name__ == "__main__":
    main()
