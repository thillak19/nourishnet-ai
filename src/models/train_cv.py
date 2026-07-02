from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES = ["pizza", "sushi", "ice_cream", "samosa", "fried_rice", "donuts", "omelette"]
EPOCHS = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_subset(dataset, selected_indices):
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "_labels", None)
    if targets is None:
        raise AttributeError("Dataset does not expose targets or _labels")
    targets = np.array(targets)
    mask = np.isin(targets, selected_indices)
    return Subset(dataset, np.where(mask)[0])


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def train_epoch(model, loader, label_map):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        remapped = torch.tensor([label_map[label.item()] for label in labels], dtype=torch.long, device=DEVICE)
        images = images.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, remapped)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * remapped.size(0)
        correct += outputs.argmax(1).eq(remapped).sum().item()
        total += remapped.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, label_map):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            remapped = torch.tensor([label_map[label.item()] for label in labels], dtype=torch.long, device=DEVICE)
            images = images.to(DEVICE)

            outputs = model(images)
            correct += outputs.argmax(1).eq(remapped).sum().item()
            total += remapped.size(0)

    return 100.0 * correct / total


def main():
    print(f"Device: {DEVICE}")

    train_dataset = torchvision.datasets.Food101(
        root=str(DATA_DIR),
        split="train",
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.Food101(
        root=str(DATA_DIR),
        split="test",
        download=True,
        transform=transform,
    )

    class_to_idx = train_dataset.class_to_idx
    available_classes = [c for c in SELECTED_CLASSES if c in class_to_idx]
    if not available_classes:
        raise RuntimeError("No selected classes were found in the Food101 dataset.")

    selected_indices = [class_to_idx[c] for c in available_classes]
    label_map = {class_idx: idx for idx, class_idx in enumerate(selected_indices)}

    train_loader = DataLoader(
        get_subset(train_dataset, selected_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        get_subset(test_dataset, selected_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(len(available_classes))
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = train_epoch(model, train_loader, label_map)
        test_acc = evaluate(model, test_loader, label_map)
        scheduler.step()

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_food_classifier.pth")

    print(f"\nBest Accuracy: {best_acc:.2f}% | Model saved.")


if __name__ == "__main__":
    main()