import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import DeepfakeDataset
from src.models import FrequencyOnlyModel
from src.evaluate import evaluate_model


ALL_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.float().to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


def run_experiment():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    df = pd.read_csv("dataset_index.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    results = []

    for held_out in ALL_METHODS:

        train_methods = [m for m in ALL_METHODS if m != held_out]

        print("\n" + "=" * 60)
        print("Held out:", held_out)
        print("=" * 60)

        train_df = df[
            (df["split"] == "train")
            &
            (
                (df["method"].isin(train_methods))
                |
                (df["method"] == "Real")
            )
        ]

        val_df = df[
            (df["split"] == "val")
            &
            (
                (df["method"].isin(train_methods))
                |
                (df["method"] == "Real")
            )
        ]

        test_df = df[
            (df["split"] == "test")
            &
            (
                (df["method"] == held_out)
                |
                (df["method"] == "Real")
            )
        ]

        train_loader = DataLoader(
            DeepfakeDataset(train_df, transform=transform),
            batch_size=64,
            shuffle=True,
            num_workers=2,
        )

        val_loader = DataLoader(
            DeepfakeDataset(val_df, transform=transform),
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

        test_loader = DataLoader(
            DeepfakeDataset(test_df, transform=transform),
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

        model = FrequencyOnlyModel(
            transform_type="fft"
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0

        ckpt_path = f"results/checkpoints/freqonly_{held_out}.pt"

        for epoch in range(10):

            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )

            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
            )

            print(
                f"Epoch {epoch+1} "
                f"loss={loss:.4f} "
                f"val_auc={val_metrics['auc']:.4f}"
            )

            if val_metrics["auc"] > best_auc:

                best_auc = val_metrics["auc"]

                torch.save(
                    model.state_dict(),
                    ckpt_path,
                )

        model.load_state_dict(torch.load(ckpt_path))

        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
        )

        print(test_metrics)

        results.append({
            "held_out": held_out,
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "auc": test_metrics["auc"],
        })

        out_path = (
            f"results/metrics/"
            f"freqonly_holdout_{held_out}.json"
        )

        with open(out_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

    summary_df = pd.DataFrame(results)

    summary_df.to_csv(
        "results/metrics/freqonly_summary.csv",
        index=False,
    )

    print(summary_df)


if __name__ == "__main__":
    run_experiment()
