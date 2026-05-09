import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification.

    Drop-in replacement for nn.BCEWithLogitsLoss. Down-weights easy examples
    and focuses training on hard ones — useful when the model needs to
    generalize to unseen manipulation methods.

    Args:
        alpha: weight for the positive (fake) class. Values < 0.5 reduce the
               contribution of easy fake examples. Default 0.25.
        gamma: focusing parameter. Higher values down-weight easy examples
               more aggressively. Default 2.0 (standard from the original paper).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: raw model output (before sigmoid), shape (N,)
        # targets: float binary labels, shape (N,)
        targets = targets.float()

        # Standard BCE loss per element (no reduction yet)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t is the probability of the true class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t weights positive vs negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulation: down-weight easy examples
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss