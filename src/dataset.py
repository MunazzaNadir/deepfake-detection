import io
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class JPEGCompression:
    """Simulate JPEG compression artifacts by re-encoding at a random quality."""
    def __init__(self, quality_low=40, quality_high=90):
        self.quality_low = quality_low
        self.quality_high = quality_high

    def __call__(self, img):
        quality = random.randint(self.quality_low, self.quality_high)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def get_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            # RandomResizedCrop instead of plain Resize:
            # forces robustness to scale and position changes
            transforms.RandomResizedCrop(
                size=img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # Stronger ColorJitter to reduce reliance on color/lighting cues
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            # GaussianBlur simulates soft/blurry deepfake artifacts
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=0.3,
            ),
            # RandomGrayscale reduces over-reliance on color cues
            transforms.RandomGrayscale(p=0.05),
            # JPEG compression simulation — important since deepfakes are often
            # distributed as compressed images
            transforms.RandomApply(
                [JPEGCompression(quality_low=40, quality_high=90)],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.25),
        ])

    # Val/test transforms stay clean — no augmentation
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class DeepfakeImageDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split,
        methods=None,
        transform=None,
        max_samples=None,
    ):
        self.df = pd.read_csv(csv_path)

        self.df = self.df[self.df["split"] == split].copy()

        if methods is not None:
            keep_methods = ["Real"] + list(methods)
            self.df = self.df[self.df["method"].isin(keep_methods)].copy()

        if max_samples is not None:
            self.df = self.df.sample(
                n=min(max_samples, len(self.df)),
                random_state=42,
            ).reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        method = row["method"]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "method": method,
            "path": row["path"],
        }