from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_FILE = DATA_DIR / "sentiment_reviews.csv"
MODEL_DIR = ROOT_DIR / "models" / "saved" / "sentiment_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 5
BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

POSITIVE_REVIEWS = [
    "Food was absolutely delicious and delivery was super fast!",
    "Amazing taste, fresh ingredients, will order again.",
    "Best biryani I have ever had, highly recommended.",
    "Packaging was great and food arrived hot.",
    "Excellent service and the pizza was perfect.",
]

NEGATIVE_REVIEWS = [
    "Food arrived cold and was completely tasteless.",
    "Very late delivery, food was soggy and bad.",
    "Worst experience ever, never ordering from here again.",
    "Packaging was torn and food was spilled inside.",
    "Stale food delivered, made me sick afterwards.",
]

NEUTRAL_REVIEWS = [
    "Food was okay, nothing special but edible.",
    "Average experience, delivery was on time.",
    "Decent food but portion size could be better.",
    "Neither good nor bad, just regular food.",
    "Okay for the price, would not strongly recommend.",
]


def generate_dataset() -> pd.DataFrame:
    records = []
    for _ in range(600):
        records.append({"review": random.choice(POSITIVE_REVIEWS), "label": 2})
    for _ in range(600):
        records.append({"review": random.choice(NEGATIVE_REVIEWS), "label": 0})
    for _ in range(400):
        records.append({"review": random.choice(NEUTRAL_REVIEWS), "label": 1})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    return df


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return generate_dataset()

    df = pd.read_csv(DATA_FILE)
    if "review" not in df.columns or "label" not in df.columns:
        raise ValueError("sentiment_reviews.csv must contain 'review' and 'label' columns")
    return df


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def main():
    print(f"Device: {DEVICE}")

    df = load_data()
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        ReviewDataset(train_df["review"], train_df["label"], tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        ReviewDataset(test_df["review"], test_df["label"], tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

    with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"\nBest Accuracy: {best_acc:.2f}% | Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()