
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

# -------------------- SETTINGS --------------------

ROOT = Path(
    r"D:\AI_FINAL_PROJECT\ThermalImagesDataset\ThermalImagesDataset-main"
)
DATA_DIR = ROOT / "Sources" / "lstm" / "data" / "sequences"
OUT_DIR = Path(r"D:\AI_FINAL_PROJECT\outputs\thermal_sequence_models")

MAX_LEN = 32
BATCH_SIZE = 128
EPOCHS = 8
PATIENCE = 2
SEED = 42

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- LOAD DATA --------------------

def load_sequences():
    X, y, groups, filenames = [], [], [], []
    bad_files = 0

    for folder_name, folder_label in [
        ("normal", 0),
        ("yawning", 1),
    ]:
        folder = DATA_DIR / folder_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        files = sorted(folder.glob("*.txt"))

        print(f"\nReading {folder_name}: {len(files)} files")

        for i, path in enumerate(files, start=1):
            try:
                arr = np.loadtxt(path, delimiter=";", dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                # Expected format: 3 features + 1 provisional frame label.
                if arr.ndim != 2 or arr.shape[1] != 4:
                    bad_files += 1
                    continue

                if arr.shape[0] == 0 or not np.isfinite(arr).all():
                    bad_files += 1
                    continue

                frame_labels = arr[:, 3]

                # Do not silently accept unexpected label values.
                if not np.isin(frame_labels, [0.0, 1.0]).all():
                    bad_files += 1
                    continue

                features = arr[:, :3]
                sequence_label = int(frame_labels.mean() >= 0.5)

                # Pad short sequences by repeating their last frame.
                # Truncate only if a sequence is longer than MAX_LEN.
                length = min(len(features), MAX_LEN)
                features = features[:MAX_LEN]

                if length < MAX_LEN:
                    padding = np.repeat(
                        features[-1:, :], MAX_LEN - length, axis=0
                    )
                    features = np.concatenate([features, padding], axis=0)

                X.append(features)
                y.append(sequence_label)

                # Keep overlapping windows from the same recording together.
                group = path.stem.split("_face_")[0]
                groups.append(group)
                filenames.append(path.name)

            except Exception:
                bad_files += 1

            if i % 5000 == 0:
                print(f"  Read {i}/{len(files)}")

    if not X:
        raise RuntimeError("No valid sequence files were loaded.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    filenames = np.asarray(filenames)

    print("\n------------ DATA SUMMARY ------------")
    print("Valid sequences:", len(X))
    print("Skipped files:", bad_files)
    print("Input shape:", X.shape)
    print("Sequence labels (0=not yawning, 1=yawning):")
    print(pd.Series(y).value_counts().sort_index().to_dict())
    print("Unique recording groups:", len(np.unique(groups)))

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            "Both sequence classes were not found. "
            "Check whether the fourth column really contains frame labels."
        )

    # The folder is only a rough reference for checking the provisional labels.
    folder_labels = np.array([
        0 if "_Normal_" in name else 1
        for name in filenames
    ])
    disagreement = np.mean(folder_labels != y)
    print(
        "Folder-name vs derived-label disagreement: "
        f"{disagreement:.2%}"
    )
    print(
        "IMPORTANT: Fourth-column labels are an assumption, "
        "not yet independently verified."
    )

    return X, y, groups, filenames


# -------------------- GROUP-WISE SPLIT --------------------

def make_splits(y, groups):
    # Approximately 70% train, 15% validation, 15% test.
    # GroupShuffleSplit prevents overlapping windows from the same
    # recording appearing across multiple splits.
    for seed in range(SEED, SEED + 100):
        first = GroupShuffleSplit(
            n_splits=1, test_size=0.30, random_state=seed
        )
        train_idx, temp_idx = next(
            first.split(np.zeros(len(y)), y, groups)
        )

        second = GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=seed + 1000
        )
        val_rel, test_rel = next(
            second.split(
                np.zeros(len(temp_idx)),
                y[temp_idx],
                groups[temp_idx],
            )
        )

        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        if all(len(np.unique(y[idx])) == 2
               for idx in (train_idx, val_idx, test_idx)):
            break
    else:
        raise RuntimeError(
            "Could not make group-wise splits containing both classes. "
            "No random window-level split was used because it could leak "
            "overlapping sequences between train and test."
        )

    for name, idx in [
        ("Train", train_idx),
        ("Validation", val_idx),
        ("Test", test_idx),
    ]:
        print(
            f"{name}: {len(idx)} sequences; "
            f"labels={np.bincount(y[idx], minlength=2).tolist()}; "
            f"groups={len(np.unique(groups[idx]))}"
        )

    return train_idx, val_idx, test_idx


# -------------------- NORMALIZE FEATURES --------------------

def normalize_features(X, train_idx):
    # Fit statistics on training sequences only.
    train_frames = X[train_idx].reshape(-1, X.shape[-1])
    mean = train_frames.mean(axis=0)
    std = train_frames.std(axis=0)
    std[std < 1e-6] = 1.0

    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean, std


class SequenceDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- MODELS --------------------

class TemporalCNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        return self.classifier(x.mean(dim=1))


class TemporalCNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=32,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.classifier(x.mean(dim=1))


class TemporalCNN_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


# -------------------- TRAIN AND EVALUATE --------------------

def evaluate(model, loader, criterion):
    model.eval()
    losses, all_y, all_pred = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            losses.append(criterion(logits, yb).item())
            all_y.extend(yb.numpy().tolist())
            all_pred.extend(logits.argmax(dim=1).numpy().tolist())

    return (
        float(np.mean(losses)),
        np.asarray(all_y),
        np.asarray(all_pred),
    )


def train_model(name, model_class, train_loader, val_loader,
                X, y, train_idx, val_idx, test_idx,
                filenames, mean, std):

    print(f"\n\n========== TRAINING {name} ==========")
    model = model_class().to(DEVICE)

    train_counts = np.bincount(y[train_idx], minlength=2)
    weights = len(train_idx) / (
        2.0 * np.maximum(train_counts, 1)
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    best_f1 = -1.0
    best_state = None
    wait = 0
    start_time = time.time()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        val_loss, val_y, val_pred = evaluate(
            model, val_loader, criterion
        )
        val_f1 = f1_score(
            val_y, val_pred, average="macro", zero_division=0
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "validation_loss": val_loss,
            "validation_macro_f1": float(val_f1),
        })

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss={np.mean(batch_losses):.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val macro-F1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint saved for {name}")

    model.load_state_dict(best_state)

    model_path = OUT_DIR / f"{name}.pt"
    torch.save({
        "model_name": name,
        "model_state_dict": model.state_dict(),
        "input_features": 3,
        "sequence_length": MAX_LEN,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "label_rule": "majority of fourth-column binary frame labels",
        "best_validation_macro_f1": float(best_f1),
    }, model_path)

    # Test set is evaluated only after model selection using validation F1.
    test_loader = DataLoader(
        SequenceDataset(X, y, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    _, test_y, test_pred = evaluate(model, test_loader, criterion)

    report = classification_report(
        test_y,
        test_pred,
        labels=[0, 1],
        target_names=["not_yawning", "yawning"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_y, test_pred, labels=[0, 1])

    metrics = {
        "model": name,
        "test_sequences": int(len(test_idx)),
        "accuracy": float(accuracy_score(test_y, test_pred)),
        "macro_f1": float(f1_score(
            test_y, test_pred, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            test_y, test_pred, average="weighted", zero_division=0
        )),
        "precision_macro": float(precision_score(
            test_y, test_pred, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            test_y, test_pred, average="macro", zero_division=0
        )),
        "confusion_matrix_labels_order": [
            "not_yawning", "yawning"
        ],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "best_validation_macro_f1": float(best_f1),
        "training_seconds": round(time.time() - start_time, 2),
        "checkpoint": str(model_path),
        "history": history,
    }

    with open(
        OUT_DIR / f"{name}_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "filename": filenames[test_idx],
        "actual_label": test_y,
        "predicted_label": test_pred,
    }).to_csv(
        OUT_DIR / f"{name}_test_predictions.csv",
        index=False,
    )

    print(f"\n{name} TEST RESULTS")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(cm)
    print("Saved:", model_path)

    return metrics


def main():
    print("Device:", DEVICE)
    print("Loading thermal sequence files...")
    X, y, groups, filenames = load_sequences()

    train_idx, val_idx, test_idx = make_splits(y, groups)
    X, mean, std = normalize_features(X, train_idx)

    # Save the split manifest so the experiment is reproducible.
    split_rows = []
    for split_name, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        for idx in indices:
            split_rows.append({
                "filename": filenames[idx],
                "group": groups[idx],
                "label": int(y[idx]),
                "split": split_name,
            })

    pd.DataFrame(split_rows).to_csv(
        OUT_DIR / "split_manifest.csv", index=False
    )

    train_loader = DataLoader(
        SequenceDataset(X, y, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SequenceDataset(X, y, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    models = [
        ("temporal_cnn_bilstm", TemporalCNN_BiLSTM),
        ("temporal_cnn_gru", TemporalCNN_GRU),
        ("temporal_cnn_transformer", TemporalCNN_Transformer),
    ]

    results = []
    for name, model_class in models:
        result = train_model(
            name, model_class, train_loader, val_loader,
            X, y, train_idx, val_idx, test_idx,
            filenames, mean, std,
        )
        results.append(result)

    summary = pd.DataFrame([{
        "model": r["model"],
        "accuracy": r["accuracy"],
        "macro_f1": r["macro_f1"],
        "weighted_f1": r["weighted_f1"],
        "precision_macro": r["precision_macro"],
        "recall_macro": r["recall_macro"],
        "training_seconds": r["training_seconds"],
    } for r in results]).sort_values(
        "macro_f1", ascending=False
    )

    summary.to_csv(OUT_DIR / "thermal_model_comparison.csv", index=False)

    print("\n\n========== FINAL COMPARISON ==========")
    print(summary.to_string(index=False))
    print("\nAll outputs saved in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
