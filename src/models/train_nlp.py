"""
NourishNet AI - Sentiment Analysis
Fine-tuning DistilBERT for restaurant review sentiment classification.
"""

import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)

positive_reviews = [
    "Food was absolutely delicious and delivery was super fast!",
    "Amazing taste, fresh ingredients, will order again.",
    "Best biryani I have ever had, highly recommended.",
    "Packaging was great and food arrived hot.",
    "Excellent service and the pizza was perfect.",
]
negative_reviews = [
    "Food arrived cold and was completely tasteless.",
    "Very late delivery, food was soggy and bad.",
    "Worst experience ever, never ordering from here again.",
    "Packaging was torn and food was spilled inside.",
    "Stale food delivered, made me sick afterwards.",
]
neutral_reviews = [
    "Food was okay, nothing special but edible.",
    "Average experience, delivery was on time.",
    "Decent food but portion size could be better.",
    "Neither good nor bad, just regular food.",
    "Okay for the price, would not strongly recommend.",
]

def generate_dataset():
    reviews, labels = [], []
    for _ in range(600):
        reviews.append(random.choice(positive_reviews) + " " + random.choice(positive_reviews))
        labels.append(2)
    for _ in range(600):
        reviews.append(random.choice(negative_reviews) + " " + random.choice(negative_reviews))
        labels.append(0)
    for _ in range(400):
        reviews.append(random.choice(neutral_reviews) + " " + random.choice(neutral_reviews))
        labels.append(1)
    df = pd.DataFrame({"review": reviews, "sentiment": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=128, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += out.logits.argmax(1).eq(lbls).sum().item()
        total   += lbls.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            correct += out.logits.argmax(1).eq(lbls).sum().item()
            total   += lbls.size(0)
    return 100. * correct / total

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    df = generate_dataset()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(ReviewDataset(train_df["review"], train_df["sentiment"], tokenizer), batch_size=32, shuffle=True)
    test_loader  = DataLoader(ReviewDataset(test_df["review"],  test_df["sentiment"],  tokenizer), batch_size=32)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            model.save_pretrained(MODEL_DIR / "sentiment_model")
            tokenizer.save_pretrained(MODEL_DIR / "sentiment_model")

    print(f"\nBest Accuracy: {best_acc:.2f}% | Model saved.")