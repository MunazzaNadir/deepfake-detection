
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    all_probs = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "macro_f1": f1_score(all_labels, preds, average="macro"),
        "auc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, preds).tolist(),
    }

    return metrics
