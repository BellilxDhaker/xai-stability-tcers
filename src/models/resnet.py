from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model: nn.Module, train_loader, validation_loader, device: torch.device,
                epochs: int = 10, learning_rate: float = 3e-4,
                weight_decay: float = 1e-4, patience: int = 3,
                checkpoint: Path | None = None) -> List[Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=1)
    model.to(device)
    best_loss = float("inf")
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_count = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_count += labels.size(0)
        validation_loss, validation_correct, validation_count = evaluate_loss(
            model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_count, 1),
            "train_accuracy": train_correct / max(train_count, 1),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_correct / max(validation_count, 1),
        }
        history.append(row)
        print(
            f"epoch {row['epoch']:02d} | train_loss {row['train_loss']:.4f} | "
            f"train_acc {row['train_accuracy']:.4f} | val_loss {row['validation_loss']:.4f} | "
            f"val_acc {row['validation_accuracy']:.4f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "history": history}, checkpoint)
                print(f"  saved checkpoint (best val_loss {best_loss:.4f})", flush=True)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return history


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    loss_total, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_total += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    return loss_total / max(count, 1), correct, count


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    labels, predictions, positive_scores = [], [], []
    for images, batch_labels in loader:
        logits = model(images.to(device))
        probabilities = logits.softmax(1)
        labels.extend(batch_labels.numpy().tolist())
        predictions.extend(probabilities.argmax(1).cpu().numpy().tolist())
        positive_scores.extend(probabilities[:, 1].cpu().numpy().tolist())
    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    metrics["roc_auc"] = roc_auc_score(labels, positive_scores) if len(np.unique(labels)) == 2 else float("nan")
    return metrics
