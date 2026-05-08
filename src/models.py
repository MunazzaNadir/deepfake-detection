
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

import torch
import torch.nn as nn
import torch.fft


# ─────────────────────────────────────────────────────────────
# FFT FEATURES
# ─────────────────────────────────────────────────────────────

def fft2d_log(x):
    freq = torch.fft.fft2(x)
    freq_shifted = torch.fft.fftshift(freq)
    magnitude = torch.abs(freq_shifted)

    log_magnitude = torch.log1p(magnitude)

    b = log_magnitude.shape[0]
    mn = log_magnitude.view(b, -1).min(1)[0].view(b, 1, 1, 1)
    mx = log_magnitude.view(b, -1).max(1)[0].view(b, 1, 1, 1)

    return (log_magnitude - mn) / (mx - mn + 1e-8)


# ─────────────────────────────────────────────────────────────
# DCT FEATURES
# ─────────────────────────────────────────────────────────────

def dct2d_torch(x):

    def dct1d(signal):
        n = signal.shape[-1]

        v = torch.cat([signal, signal.flip(-1)], dim=-1)

        V = torch.fft.rfft(v, dim=-1)

        k = torch.arange(n, device=signal.device, dtype=signal.dtype)

        W = torch.exp(-1j * torch.pi * k / (2 * n))

        return (V[..., :n] * W).real

    b, c, h, w = x.shape

    out = dct1d(x.reshape(b * c, h, w))

    out = dct1d(out.transpose(-1, -2)).transpose(-1, -2)

    out = torch.log1p(torch.abs(out))

    mn = out.view(b, -1).min(1)[0].view(b, 1, 1, 1)
    mx = out.view(b, -1).max(1)[0].view(b, 1, 1, 1)

    out = (out - mn) / (mx - mn + 1e-8)

    return out.view(b, c, h, w)


# ─────────────────────────────────────────────────────────────
# SHARED FREQUENCY CNN
# ─────────────────────────────────────────────────────────────

class FrequencyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
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

            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# FREQUENCY ONLY MODEL
# ─────────────────────────────────────────────────────────────

class FrequencyOnlyModel(nn.Module):

    def __init__(self, transform_type="fft"):
        super().__init__()

        assert transform_type in ["fft", "dct"]

        self.transform_type = transform_type

        self.freq_cnn = FrequencyCNN()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def compute_frequency(self, x):

        if self.transform_type == "fft":
            return fft2d_log(x)

        return dct2d_torch(x)

    def forward(self, x):

        freq = self.compute_frequency(x)

        feats = self.freq_cnn(freq)

        feats = feats.view(feats.size(0), -1)

        logits = self.classifier(feats)

        return logits.squeeze(1)
