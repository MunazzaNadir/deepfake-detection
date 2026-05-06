
import torch
import torch.nn as nn
import torch.fft
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetBaseline(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits.squeeze(1)


def fft_log_magnitude(x):
    """
    Approximate frequency representation using 2D FFT log magnitude.
    Input: x of shape [B, 3, H, W]
    Output: log magnitude frequency map of shape [B, 3, H, W]
    """
    freq = torch.fft.fft2(x, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    mag = torch.abs(freq)
    log_mag = torch.log1p(mag)
    return log_mag


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        freq = fft_log_magnitude(x)
        features = self.cnn(freq)
        features = features.flatten(1)
        features = self.proj(features)
        return features


class DualBranchSpatialFrequencyNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        efficientnet = efficientnet_b0(weights=weights)

        self.spatial_features = efficientnet.features
        self.spatial_pool = efficientnet.avgpool

        self.spatial_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.frequency_branch = FrequencyBranch(out_dim=512)

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        spatial = self.spatial_features(x)
        spatial = self.spatial_pool(spatial)
        spatial = self.spatial_proj(spatial)

        frequency = self.frequency_branch(x)

        fused = torch.cat([spatial, frequency], dim=1)
        logits = self.fusion(fused)

        return logits.squeeze(1)
