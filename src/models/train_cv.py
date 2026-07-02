"""
NourishNet AI - Food Image Classification
Transfer Learning with MobileNetV2 (PyTorch)
Run this on a machine with GPU for best performance.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES = ['pizza', 'sushi', 'ice_cream', 'samosa', 'fried_rice', 'donuts', 'omelette']
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_subset(dataset, selected_indices):
    targets = np.array(dataset._labels)
    mask = np.isin(targets, selected_indices)
    return Subset(dataset, np.where(mask)[0])

def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)

def train(model, loader, optimizer, criterion, selected_idx_values):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        remapped = torch.tensor([selected_idx_values.index(l.item()) for l in labels])
        images, remapped = images.to(DEVICE), remapped.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, remapped)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += outputs.argmax(1).eq(remapped).sum().item()
        total += remapped.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, selected_idx_values):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            remapped = torch.tensor([selected_idx_values.index(l.item()) for l in labels])
            images, remapped = images.to(DEVICE), remapped.to(DEVICE)
            correct += model(images).argmax(1).eq(remapped).sum().item()
            total += remapped.size(0)
    return 100. * correct / total

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    train_dataset = torchvision.datasets.Food101(root="./data", split="train", download=True, transform=transform)
    test_dataset  = torchvision.datasets.Food101(root="./data", split="test",  download=True, transform=transform)

    class_to_idx = train_dataset.class_to_idx
    available    = [c for c in SELECTED_CLASSES if c in class_to_idx]
    idx_values   = [class_to_idx[c] for c in available]

    train_loader = DataLoader(get_subset(train_dataset, idx_values), batch_size=32, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(get_subset(test_dataset,  idx_values), batch_size=32, shuffle=False, num_workers=2)

    model     = build_model(len(available))
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        loss, train_acc = train(model, train_loader, optimizer, criterion, idx_values)
        test_acc = evaluate(model, test_loader, idx_values)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_food_classifier.pth")

    print(f"\nBest Accuracy: {best_acc:.2f}% | Model saved.")