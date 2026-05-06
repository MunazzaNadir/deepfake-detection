
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import DeepfakeImageDataset, get_transforms
from src.models import EfficientNetBaseline, DualBranchSpatialFrequencyNet


def load_model(model_type, checkpoint_path, device):
    if model_type == "baseline":
        model = EfficientNetBaseline()
    elif model_type == "dual":
        model = DualBranchSpatialFrequencyNet()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, dataloader, device):
    all_probs, all_labels, all_methods, all_paths = [], [], [], []

    for batch in dataloader:
        images = batch["image"].to(device)
        probs = torch.sigmoid(model(images)).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch["label"].numpy())
        all_methods.extend(batch["method"])
        all_paths.extend(batch["path"])

    return (
        np.array(all_probs),
        np.array(all_labels),
        np.array(all_methods),
        np.array(all_paths),
    )


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Real", "Pred Fake"])
    ax.set_yticklabels(["True Real", "True Fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, color="black")
    ax.set_title("Confusion Matrix — FaceSwap vs Real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {output_path}")


def save_false_negatives(paths, output_dir, n=10):
    fn_dir = output_dir / "false_negatives"
    fn_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths[:n]):
        img = Image.open(path).convert("RGB")
        img.save(fn_dir / f"fn_{i:02d}_{Path(path).name}")
    print(f"Saved {min(n, len(paths))} false negative images → {fn_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--csv_path", default="dataset_index.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_type", required=True, choices=["baseline", "dual"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.model_type, args.checkpoint_path, device)

    # DeepfakeImageDataset with methods=["FaceSwap"] keeps Real + FaceSwap samples
    dataset = DeepfakeImageDataset(
        csv_path=args.csv_path,
        split="test",
        methods=["FaceSwap"],
        transform=get_transforms(train=False),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nRunning inference on {len(dataset)} samples (FaceSwap + Real test set)...")
    probs, labels, methods, paths = run_inference(model, loader, device)

    faceswap_mask = methods == "FaceSwap"
    real_mask = methods == "Real"

    print(f"\nAverage predicted fake probability:")
    print(f"  FaceSwap : {probs[faceswap_mask].mean():.4f}  (n={faceswap_mask.sum()})")
    print(f"  Real     : {probs[real_mask].mean():.4f}  (n={real_mask.sum()})")

    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    fn_mask = faceswap_mask & (preds == 0)
    fn_paths = paths[fn_mask]
    print(f"\nFaceSwap false negatives: {fn_mask.sum()} / {faceswap_mask.sum()}")
    save_false_negatives(fn_paths, output_dir)


if __name__ == "__main__":
    main()
