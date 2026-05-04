
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import DeepfakeImageDataset, get_transforms
from src.models import EfficientNetBaseline
from src.evaluate import evaluate_model


METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_weighted_sampler(dataset):
    labels = dataset.df["label"].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_loader.dataset)


def run_experiment(
    experiment_name,
    csv_path="dataset_index.csv",
    train_methods=None,
    val_methods=None,
    test_methods=None,
    batch_size=32,
    epochs=5,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
    lr=1e-4,
    weight_decay=1e-4,
    seed=42,
):
    set_seed(seed)

    train_methods = train_methods or METHODS
    val_methods = val_methods or train_methods
    test_methods = test_methods or METHODS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    results_dir = Path("results")
    checkpoint_dir = results_dir / "checkpoints"
    metrics_dir = results_dir / "metrics"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DeepfakeImageDataset(
        csv_path=csv_path,
        split="train",
        methods=train_methods,
        transform=get_transforms(train=True),
        max_samples=max_train_samples,
    )

    val_ds = DeepfakeImageDataset(
        csv_path=csv_path,
        split="val",
        methods=val_methods,
        transform=get_transforms(train=False),
        max_samples=max_val_samples,
    )

    test_ds = DeepfakeImageDataset(
        csv_path=csv_path,
        split="test",
        methods=test_methods,
        transform=get_transforms(train=False),
        max_samples=max_test_samples,
    )

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Test size:", len(test_ds))

    print("\nTrain label counts:")
    print(train_ds.df["label"].value_counts())

    print("\nTrain method counts:")
    print(train_ds.df["method"].value_counts())

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = EfficientNetBaseline().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_checkpoint_path = checkpoint_dir / f"{experiment_name}.pt"

    history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate_model(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_auc": float(val_metrics["auc"]),
        }

        history.append(row)
        print(row)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print("Saved best checkpoint:", best_checkpoint_path)

    print("\nLoading best checkpoint...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    test_metrics = evaluate_model(model, test_loader, device)

    output = {
        "experiment_name": experiment_name,
        "train_methods": train_methods,
        "val_methods": val_methods,
        "test_methods": test_methods,
        "best_val_auc": float(best_val_auc),
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "auc": float(test_metrics["auc"]),
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "history": history,
    }

    metrics_path = metrics_dir / f"{experiment_name}.json"

    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nTest metrics:")
    print(output["test_metrics"])
    print("Saved metrics:", metrics_path)

    return output


if __name__ == "__main__":
    run_experiment(
        experiment_name="baseline_small",
        epochs=3,
        max_train_samples=5000,
        max_val_samples=1500,
        max_test_samples=1500,
    )
