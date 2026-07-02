from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.generate_dataset import generate_dataset
from src.utils.helpers import ensure_dir, save_pickle

DATA_PATH = ROOT_DIR / "data" / "raw" / "orders.csv"
MODEL_DIR = ROOT_DIR / "models" / "saved"
RECOMMENDATION_DIR = MODEL_DIR / "recommendation_models"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("Dataset not found. Generating it now...")
        generate_dataset(2000, DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str] | None = None) -> ColumnTransformer:
    transformers = [("num", SimpleImputer(strategy="median"), numeric_features)]

    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    return ColumnTransformer(transformers=transformers)


def train_delivery_model(df: pd.DataFrame):
    features = [
        "distance_km",
        "restaurant_prep_time_min",
        "hour_of_day",
        "is_peak_hour",
        "order_value_inr",
        "customer_total_past_orders",
        "customer_is_premium",
        "customer_complaints_90d",
        "cuisine_type",
        "weather",
        "traffic_level",
        "vehicle_type",
    ]
    X = df[features]
    y = df["delivery_time_min"]

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    numeric_features=[
                        "distance_km",
                        "restaurant_prep_time_min",
                        "hour_of_day",
                        "is_peak_hour",
                        "order_value_inr",
                        "customer_total_past_orders",
                        "customer_is_premium",
                        "customer_complaints_90d",
                    ],
                    categorical_features=["cuisine_type", "weather", "traffic_level", "vehicle_type"],
                ),
            ),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    mae = mean_absolute_error(y, preds)
    save_pickle(pipeline, MODEL_DIR / "delivery_time_best_classical.pkl")
    print(f"Saved delivery model -> {MODEL_DIR / 'delivery_time_best_classical.pkl'}")
    print(f"Delivery MAE: {mae:.2f}")
    return pipeline


def train_churn_model(df: pd.DataFrame):
    features = [
        "distance_km",
        "delivery_time_min",
        "customer_rating",
        "customer_total_past_orders",
        "customer_complaints_90d",
        "order_value_inr",
        "customer_is_premium",
    ]
    X = df[features]
    y = df["churned"]

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features=features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    save_pickle(pipeline, MODEL_DIR / "churn_best_model.pkl")
    print(f"Saved churn model -> {MODEL_DIR / 'churn_best_model.pkl'}")
    print(f"Churn accuracy: {acc:.3f}")
    return pipeline


def train_recommendation_artifacts(df: pd.DataFrame):
    ensure_dir(RECOMMENDATION_DIR)

    user_item_matrix = df.pivot_table(
        index="user_id",
        columns="food_name",
        values="rating",
        aggfunc="mean",
        fill_value=0,
    ).astype(float)

    user_similarity = pd.DataFrame(
        cosine_similarity(user_item_matrix),
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    item_matrix = user_item_matrix.T
    item_similarity = pd.DataFrame(
        cosine_similarity(item_matrix),
        index=item_matrix.index,
        columns=item_matrix.index,
    )

    save_pickle(user_item_matrix, RECOMMENDATION_DIR / "user_item_matrix.pkl")
    save_pickle(user_similarity, RECOMMENDATION_DIR / "user_similarity.pkl")
    save_pickle(item_similarity, RECOMMENDATION_DIR / "item_similarity.pkl")

    print(f"Saved recommendation artifacts -> {RECOMMENDATION_DIR}")
    return user_item_matrix, user_similarity, item_similarity


def main():
    ensure_dir(MODEL_DIR)
    df = load_dataset()

    train_delivery_model(df)
    train_churn_model(df)
    train_recommendation_artifacts(df)

    print("Training complete.")


if __name__ == "__main__":
    main()