"""
Quick bootstrap script to create minimal model artifacts the API expects.
This trains small/sketch models on the synthetic dataset (fast) and saves them
with the filenames the API loads:
 - models/saved/delivery_time_best_classical.pkl
 - models/saved/churn_best_model.pkl
 - models/saved/recommendation_models/user_item_matrix.pkl
 - models/saved/recommendation_models/user_similarity.pkl
 - models/saved/recommendation_models/item_similarity.pkl
 - models/saved/best_food_classifier.pth
 - models/saved/sentiment_model/  (HuggingFace format)

This is intended for local demo/testing. Artifacts are lightweight and fast to produce.
"""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "orders.csv"
MODEL_DIR = ROOT / "models" / "saved"
RECO_DIR = MODEL_DIR / "recommendation_models"

os.makedirs(RECO_DIR, exist_ok=True)

print("Loading or generating data...")
if not DATA_PATH.exists():
    print("Data not found — running data generator")
    import subprocess
    subprocess.check_call(["python", "src/data/generate_dataset.py"]) 

orders = pd.read_csv(DATA_PATH)
print(f"Orders loaded: {len(orders)} rows")
# Use a subset to make training fast
orders_sample = orders.sample(min(len(orders), 2000), random_state=42)

# ------------------ DELIVERY TIME MODEL (quick classical) ------------------
print("Training quick delivery time model...")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

NUMERIC_FEATURES = [
    "distance_km", "restaurant_prep_time_min", "hour_of_day",
    "is_peak_hour", "order_value_inr", "customer_total_past_orders",
    "customer_is_premium", "customer_complaints_90d",
]
CATEGORICAL_FEATURES = ["cuisine_type", "weather", "traffic_level", "vehicle_type"]
TARGET = "delivery_time_min"

X = orders_sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = orders_sample[TARGET]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1))
])
pipe.fit(X, y)
joblib.dump(pipe, MODEL_DIR / "delivery_time_best_classical.pkl")
print("Saved delivery_time_best_classical.pkl")

# ------------------ CHURN MODEL (quick classical) ------------------
print("Training quick churn model...")
from sklearn.ensemble import RandomForestClassifier

NUMERIC_CHURN = [
    "distance_km", "delivery_time_min", "customer_rating",
    "customer_total_past_orders", "customer_complaints_90d", "order_value_inr",
]
CAT_CHURN = ["customer_is_premium"]
TARGET_CHURN = "churned"

Xc = orders_sample[NUMERIC_CHURN + CAT_CHURN]
yc = orders_sample[TARGET_CHURN]

preproc_c = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_CHURN),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_CHURN),
])

pipe_c = Pipeline([
    ("preprocess", preproc_c),
    ("model", RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42, n_jobs=1))
])
pipe_c.fit(Xc, yc)
joblib.dump(pipe_c, MODEL_DIR / "churn_best_model.pkl")
print("Saved churn_best_model.pkl")

# ------------------ RECOMMENDATION (simple user-item matrix + similarities) ------------------
print("Creating simple recommendation artifacts...")
# Build a user-item matrix using customer_id x cuisine_type (count of orders)
user_item = orders.pivot_table(index='customer_id', columns='cuisine_type', values='order_id', aggfunc='count', fill_value=0)
from sklearn.metrics.pairwise import cosine_similarity

user_sim = pd.DataFrame(cosine_similarity(user_item), index=user_item.index, columns=user_item.index)
item_sim = pd.DataFrame(cosine_similarity(user_item.T), index=user_item.columns, columns=user_item.columns)

joblib.dump(user_item, RECO_DIR / "user_item_matrix.pkl")
joblib.dump(user_sim, RECO_DIR / "user_similarity.pkl")
joblib.dump(item_sim, RECO_DIR / "item_similarity.pkl")
print("Saved recommendation artifacts in recommendation_models/")

# ------------------ CV MODEL (lightweight random weights saved as expected .pth) ------------------
print("Creating lightweight CV model state_dict (mobilenet_v2 architecture)...")
import torch
import torchvision.models as models

cv_classes = ['donuts', 'fried_rice', 'ice_cream', 'omelette', 'pizza', 'samosa', 'sushi']

m = models.mobilenet_v2(weights=None)
# replace classifier to match API
try:
    last_channel = m.last_channel
except AttributeError:
    # older torchvision uses classifier[1].in_features
    last_channel = m.classifier[1].in_features
m.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(last_channel, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, len(cv_classes))
)
# random init is fine for demo
torch.save(m.state_dict(), MODEL_DIR / "best_food_classifier.pth")
print("Saved best_food_classifier.pth")

# ------------------ SENTIMENT MODEL (HuggingFace format, small initialized model) ------------------
print("Creating small HuggingFace-format sentiment_model (random init, compatible tokenizer+model)")
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

base = 'distilbert-base-uncased'
# download tokenizer files (small)
tokenizer = AutoTokenizer.from_pretrained(base)
config = AutoConfig.from_pretrained(base, num_labels=3)
model = AutoModelForSequenceClassification.from_config(config)

sent_dir = MODEL_DIR / 'sentiment_model'
sent_dir.mkdir(parents=True, exist_ok=True)

# save both tokenizer and model in HF format so src/api/main.py can load from this folder
tokenizer.save_pretrained(sent_dir)
model.save_pretrained(sent_dir)
print(f"Saved sentiment_model in {sent_dir}")

print("All artifacts created successfully.")
