# fair_evaluate_models.py
# Evaluates existing trained checkpoints without retraining or touching the test split.
# Run from D:\AI_FINAL_PROJECT with the project's .venv Python.

import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\AI_FINAL_PROJECT")
DATASET = ROOT / "outputs" / "yolov12_cnn_v2" / "dataset_v2"
VAL_DIR = DATASET / "val"
OUT = ROOT / "outputs" / "fair_evaluation_v1"
DEVICE = torch.device("cpu")
BATCH_SIZE = 4  # CPU-safe; EfficientNet + ViT feature extraction may take a while.

# Keep model outputs separate from existing experiment outputs.
OUT.mkdir(parents=True, exist_ok=True)
if not VAL_DIR.is_dir():
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

val_ds = datasets.ImageFolder(str(VAL_DIR))
folder_classes = list(val_ds.classes)
image_paths = [Path(p) for p, _ in val_ds.samples]
folder_targets = np.array([y for _, y in val_ds.samples], dtype=np.int64)
if len(val_ds) == 0:
    raise RuntimeError(f"No validation images found in {VAL_DIR}")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = [
    "safe_driving", "hair_makeup", "adjusting_radio", "gps_operating",
    "texting_right_hand", "texting_left_hand", "phone_right_hand",
    "phone_left_hand", "taking_picture", "talking_to_passenger",
    "singing_or_dancing", "fatigue_somnolence", "drinking_right_hand",
    "drinking_left_hand", "reaching_behind", "smoking"
]

def write_metrics(slug, model_names, y_true, y_pred, extra=None):
    model_names = list(model_names)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) != len(y_pred):
        raise ValueError(f"{slug}: prediction count {len(y_pred)} != label count {len(y_true)}")
    labels = list(range(len(model_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=model_names,
        digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    (OUT / f"classification_report_{slug}.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm, index=model_names, columns=model_names).to_csv(
        OUT / f"confusion_matrix_{slug}.csv", encoding="utf-8-sig"
    )
    summary = {
        "model": slug,
        "evaluation_protocol": "same_1024_v2_validation_images",
        "validation_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if extra:
        summary.update(extra)
    pd.DataFrame({
        "image_path": [str(p) for p in image_paths],
        "true_class": [model_names[i] for i in y_true],
        "predicted_class": [model_names[i] for i in y_pred],
        "correct": y_true == y_pred,
    }).to_csv(OUT / f"predictions_{slug}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{slug}] accuracy={summary['accuracy']*100:.2f}% | macro-F1={summary['macro_f1']:.4f}")
    return summary

def model_target_indices(output_names):
    # Map each ImageFolder class label into the model checkpoint's output ordering.
    name_to_idx = {name: i for i, name in enumerate(output_names)}
    missing = [name for name in folder_classes if name not in name_to_idx]
    if missing:
        raise ValueError(f"Checkpoint class names do not match validation folders. Missing: {missing}")
    return np.array([name_to_idx[folder_classes[t]] for t in folder_targets], dtype=np.int64)

def run_yolo():
    from ultralytics import YOLO
    checkpoint = ROOT / "outputs" / "yolov12_cnn_v2" / "runs" / "yolov12n_classification_v2" / "weights" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = YOLO(str(checkpoint))
    names_map = model.names
    if isinstance(names_map, dict):
        names = [names_map[i] for i in sorted(names_map)]
    else:
        names = list(names_map)
    y_true = model_target_indices(names)
    preds = []
    started = time.time()
    for start in range(0, len(image_paths), 16):
        batch_paths = [str(p) for p in image_paths[start:start+16]]
        results = model.predict(source=batch_paths, imgsz=224, device="cpu", verbose=False)
        for result in results:
            preds.append(int(result.probs.top1))
        if (len(preds) % 128 == 0) or len(preds) == len(image_paths):
            print(f"  YOLOv12: predicted {len(preds)}/{len(image_paths)}")
    print(f"  YOLOv12 prediction time: {(time.time()-started)/60:.1f} min")
    return write_metrics("yolov12_cnn", names, y_true, preds)

def run_densenet():
    import joblib
    checkpoint = OUT.parent / "densenet121_svm" / "best_densenet121_svm.joblib"
    cache_path = OUT.parent / "densenet121_svm" / "val_features.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    clf = joblib.load(checkpoint)
    names = list(folder_classes)  # ImageFolder class order used by feature extraction.
    y_true = model_target_indices(names)
    use_cache = False
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            keys = set(z.files)
            feat_key = "features" if "features" in keys else ("X" if "X" in keys else None)
            label_key = "labels" if "labels" in keys else ("y" if "y" in keys else None)
            if feat_key is not None and label_key is not None:
                X = z[feat_key]
                cached_y = z[label_key].astype(np.int64)
                if len(X) == len(val_ds) and len(cached_y) == len(val_ds) and np.array_equal(cached_y, folder_targets):
                    use_cache = True
    if not use_cache:
        print("  DenseNet feature cache is missing/incompatible; extracting features now.")
        net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        net.classifier = nn.Identity()
        net.eval().to(DEVICE)
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds = datasets.ImageFolder(str(VAL_DIR), transform=tfm)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        feats = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                feats.append(net(x.to(DEVICE)).cpu().numpy())
                if (i+1) % 16 == 0:
                    print(f"  DenseNet features: {(i+1)*len(x)}/{len(ds)}")
        X = np.concatenate(feats, axis=0)
    preds = clf.predict(X).astype(np.int64)
    if preds.min(initial=0) < 0 or preds.max(initial=0) >= len(names):
        raise ValueError("DenseNet SVM produced labels outside expected class-index range.")
    return write_metrics("densenet121_svm", names, y_true, preds)

class ChannelAttention(nn.Module):
    def __init__(self, channels=960, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        # Match the original checkpoint: 1x1 Conv2d layers after global average pooling.
        pooled = x.mean(dim=(2, 3), keepdim=True)
        weights = self.mlp(pooled)
        return x * weights

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        weights = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weights

class MobileNetV3Attention(nn.Module):
    # Matches checkpoint key names and tensor shapes from train_mobilenetv3_attention.py.
    def __init__(self, num_classes):
        super().__init__()
        self.features = models.mobilenet_v3_large(weights=None).features
        self.channel_attention = ChannelAttention(960, reduction=4)
        self.spatial_attention = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            im = im.convert("RGB")
            return self.transform(im)

def get_image_logits(model, loader, device=DEVICE):
    all_logits = []
    model.eval().to(device)
    with torch.no_grad():
        for i, x in enumerate(loader):
            logits = model(x.to(device))
            all_logits.append(logits.cpu().numpy())
            if (i+1) % 32 == 0 or (i+1) == len(loader):
                print(f"  batches: {min((i+1)*loader.batch_size, len(loader.dataset))}/{len(loader.dataset)}")
    return np.concatenate(all_logits, axis=0)

def run_mobilenet():
    ckpt_path = ROOT / "outputs" / "mobilenetv3_attention" / "best_mobilenetv3_attention.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = list(ckpt["class_names"])
    y_true = model_target_indices(names)
    model = MobileNetV3Attention(num_classes=len(names))
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None:
        raise KeyError("Could not find model_state_dict/state_dict in MobileNet checkpoint.")
    model.load_state_dict(state)
    tfm = transforms.Compose([
        transforms.Resize((int(ckpt.get("image_size", 192)), int(ckpt.get("image_size", 192)))),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    loader = DataLoader(ImagePathDataset(image_paths, tfm), batch_size=16, shuffle=False, num_workers=0)
    logits = get_image_logits(model, loader)
    preds = logits.argmax(axis=1)
    return write_metrics("mobilenetv3_attention", names, y_true, preds)

def run_efficientnet_vit():
    ckpt_path = ROOT / "outputs" / "efficientnetb3_vit" / "best_efficientnetb3_vit.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = list(ckpt.get("class_names", CLASS_NAMES))
    y_true = model_target_indices(names)

    # Match the original training architecture: frozen pretrained backbones,
    # EfficientNet-B3 300px input + ViT-B/16 224px input, fused 2304-d features.
    efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    efficientnet.classifier = nn.Identity()
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    vit.heads = nn.Identity()
    efficientnet.eval().to(DEVICE)
    vit.eval().to(DEVICE)

    feature_size_eff = int(ckpt.get("efficientnet_feature_size", 1536))
    feature_size_vit = int(ckpt.get("vit_feature_size", 768))
    fusion_classifier = nn.Sequential(
        nn.Linear(feature_size_eff + feature_size_vit, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, len(names)),
    )
    state = ckpt.get("classifier_state_dict", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Could not find classifier_state_dict/model_state_dict in EfficientNet+ViT checkpoint.")
    # The saved checkpoint used a wrapper whose fusion head keys were prefixed
    # with "classifier."; the current object is the Sequential head itself.
    if state and all(k.startswith("classifier.") for k in state.keys()):
        state = {k[len("classifier."):]: v for k, v in state.items()}
    fusion_classifier.load_state_dict(state)
    fusion_classifier.eval().to(DEVICE)

    tfm_eff = transforms.Compose([
        transforms.Resize((300, 300)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tfm_vit = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    class DualImageDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            with Image.open(self.paths[idx]) as im:
                im = im.convert("RGB")
                return tfm_eff(im), tfm_vit(im)
    loader = DataLoader(DualImageDataset(image_paths), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logits_list = []
    started = time.time()
    with torch.no_grad():
        for i, (xe, xv) in enumerate(loader):
            fe = efficientnet(xe.to(DEVICE))
            fv = vit(xv.to(DEVICE))
            fused = torch.cat([fe, fv], dim=1)
            logits_list.append(fusion_classifier(fused).cpu().numpy())
            done = min((i+1)*BATCH_SIZE, len(image_paths))
            if (i+1) % 16 == 0 or done == len(image_paths):
                print(f"  EfficientNet+ViT: {done}/{len(image_paths)} images ({(time.time()-started)/60:.1f} min)")
    preds = np.concatenate(logits_list, axis=0).argmax(axis=1)
    return write_metrics("efficientnetb3_vit", names, y_true, preds,
                         extra={"note": "Pretrained feature extractors recreated; evaluation only, no retraining."})

def main():
    print("=" * 70)
    print("FAIR EVALUATION — existing checkpoints only; no training")
    print(f"Validation folder: {VAL_DIR}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Folder classes: {folder_classes}")
    print(f"Output folder: {OUT}")
    print("=" * 70)
    summaries = []
    # Each model is independently caught so a missing checkpoint doesn't prevent others.
    for slug, fn in [
        ("yolov12_cnn", run_yolo),
        ("densenet121_svm", run_densenet),
        ("mobilenetv3_attention", run_mobilenet),
        ("efficientnetb3_vit", run_efficientnet_vit),
    ]:
        try:
            print(f"\n--- Evaluating {slug} ---")
            summaries.append(fn())
        except Exception as e:
            print(f"\nERROR evaluating {slug}: {type(e).__name__}: {e}")
            summaries.append({
                "model": slug,
                "evaluation_protocol": "same_1024_v2_validation_images",
                "status": "FAILED",
                "error": f"{type(e).__name__}: {e}",
            })
    # BiLSTM is sequence-based and is deliberately listed separately using its
    # already-recorded score, not mixed into the image-model ranking.
    summaries.append({
        "model": "resnet50_bilstm",
        "evaluation_protocol": "original_sequence_validation_not_directly_comparable",
        "validation_images": 128,
        "accuracy": 0.4141,
        "macro_f1": 0.4129,
        "status": "RECORDED_ORIGINAL_RESULT",
        "note": "Sequence model: 128 validation sequences of 8 frames; excluded from same-image ranking."
    })
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "fair_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "README.txt").write_text(
        "Fair evaluation output\n\n"
        "Four single-image models are evaluated on the same V2 validation folder.\n"
        "ResNet50+BiLSTM is a temporal sequence model and is listed with its original\n"
        "result separately; it is not directly comparable with single-image models.\n"
        "No training is performed and the held-out test split is not accessed.\n"
        "Existing experiment files are not overwritten.\n\n"
        "Files: fair_evaluation_summary.csv, per-model classification reports,\n"
        "confusion matrices, and prediction CSVs.\n",
        encoding="utf-8"
    )
    print("\nFinished. Summary saved to:")
    print(OUT / "fair_evaluation_summary.csv")
    print("Check any FAILED rows in fair_evaluation_summary.csv and the console output.")

if __name__ == "__main__":
    main()
