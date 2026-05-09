"""
src/train_augment.py

Supplementary training script with stronger augmentations and focal loss support.
Drop-in supplement to train.py -- does not modify any existing files.

Usage examples:

    # Run a cross-manipulation holdout with augmentations and focal loss
    from src.train_augment import run_experiment

    run_experiment(
        experiment_name="baseline_focal_holdout_FaceSwap",
        model_type="baseline",
        loss_type="focal",
        augment=True,
        train_methods=["Deepfakes", "Face2Face", "NeuralTextures"],
        test_methods=["FaceSwap"],
        epochs=10,
        max_train_samples=10000,
        max_val_samples=1500,
        max_test_samples=1500,
    )

    # Run with augmentations but keep BCE loss (to isolate augmentation effect)
    run_experiment(
        experiment_name="baseline_augment_only_holdout_FaceSwap",
        model_type="baseline",
        loss_type="bce",
        augment=True,
        train_methods=["Deepfakes", "Face2Face", "NeuralTextures"],
        test_methods=["FaceSwap"],
        epochs=10,
    )

    # Run without augmentations and without focal loss (reproduces original behavior)
    run_experiment(
        experiment_name="baseline_vanilla_holdout_FaceSwap",
        model_type="baseline",
        loss_type="bce",
        augment=False,
        train_methods=["Deepfakes", "Face2Face", "NeuralTextures"],
        test_methods=["FaceSwap"],
        epochs=10,
    )
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import DeepfakeImageDataset, get_transforms
from src.models import EfficientNetBaseline, DualBranchSpatialFrequencyNet
from src.evaluate import evaluate_model
from src.losses import FocalLoss


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
    epochs=10,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
    lr=1e-4,
    weight_decay=1e-4,
    seed=42,
    model_type="baseline",
    loss_type="bce",
    augment=False,
    focal_alpha=0.25,
    focal_gamma=2.0,
    drive_backup_dir=None,
):
    """
    Run a single training experiment with optional augmentations and focal loss.

    Args:
        experiment_name:   Name used for checkpoint and metrics filenames.
        csv_path:          Path to dataset_index.csv.
        train_methods:     List of fake methods to include in training.
                           Defaults to all four methods.
        val_methods:       List of fake methods to include in validation.
                           Defaults to train_methods.
        test_methods:      List of fake methods to include in testing.
                           Defaults to all four methods.
        batch_size:        Training batch size. Default 32.
        epochs:            Number of training epochs. Default 10.
        max_train_samples: Cap on training samples. None uses all available.
        max_val_samples:   Cap on validation samples. None uses all available.
        max_test_samples:  Cap on test samples. None uses all available.
        lr:                Learning rate. Default 1e-4.
        weight_decay:      AdamW weight decay. Default 1e-4.
        seed:              Random seed. Default 42.
        model_type:        "baseline" for EfficientNet-B0,
                           "dual" for dual spatial-frequency model.
        loss_type:         "bce" for binary cross-entropy (default),
                           "focal" for focal loss.
        augment:           If True, applies stronger augmentations during training.
                           If False (default), uses the original basic augmentations.
                           Val and test transforms are always clean regardless.
        focal_alpha:       Focal loss alpha. Default 0.25. Only used if loss_type="focal".
        focal_gamma:       Focal loss gamma. Default 2.0. Only used if loss_type="focal".
        drive_backup_dir:  Optional path to a Google Drive directory to back up
                           checkpoints and metrics after each best epoch.
                           Example: "/content/drive/MyDrive/deepfake-results"
    """
    set_seed(seed)

    train_methods = train_methods or METHODS
    val_methods   = val_methods   or train_methods
    test_methods  = test_methods  or METHODS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {model_type}")
    print(f"Loss:   {loss_type}")
    print(f"Augment: {augment}")

    results_dir    = Path("results")
    checkpoint_dir = results_dir / "checkpoints"
    metrics_dir    = results_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DeepfakeImageDataset(
        csv_path=csv_path,
        split="train",
        methods=train_methods,
        transform=get_transforms(train=True, augment=augment),
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