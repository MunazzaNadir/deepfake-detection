
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.25),
        ])

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
