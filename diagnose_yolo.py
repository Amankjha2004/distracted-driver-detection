
from pathlib import Path
import torch

root = Path(
    r"D:\AI_FINAL_PROJECT\outputs\yolov12_cnn"
    r"\runs\yolov12n_classification"
)

for name in ["best.pt", "last.pt"]:
    path = root / "weights" / name

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print(f"\n--- {name} ---")
    print("Epoch:", checkpoint.get("epoch"))
    print("Updates:", checkpoint.get("updates"))
    print("Train metrics:", checkpoint.get("train_metrics"))
    print("Train results:", checkpoint.get("train_results"))

    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        print("No model object found")
        continue

    state = model.float().state_dict()
    total = 0
    nonzero = 0

    for key, tensor in state.items():
        if torch.is_floating_point(tensor):
            total += tensor.numel()
            nonzero += torch.count_nonzero(tensor).item()

    print("Floating-point parameters:", total)
    print("Nonzero parameter values:", nonzero)
