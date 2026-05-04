
import torch
import torch.nn as nn
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
