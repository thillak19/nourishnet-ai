"""
NourishNet AI - FastAPI Service
Serves delivery time prediction, churn prediction,
sentiment analysis, and food recommendations.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="NourishNet AI API",
    description="AI-powered food delivery intelligence platform",
    version="1.0.0"
)

MODEL_DIR = Path("models/saved")

# Load models
delivery_model = joblib.load(MODEL_DIR / "delivery_time_best_classical.pkl")
churn_model    = joblib.load(MODEL_DIR / "churn_best_model.pkl")
user_item_matrix = joblib.load(MODEL_DIR / "recommendation_models/user_item_matrix.pkl")
user_similarity  = joblib.load(MODEL_DIR / "recommendation_models/user_similarity.pkl")
item_sim_df      = joblib.load(MODEL_DIR / "recommendation_models/item_similarity.pkl")


# ---------- Schemas ----------

class DeliveryRequest(BaseModel):
    distance_km: float
    restaurant_prep_time_min: float
    hour_of_day: int
    is_peak_hour: int
    order_value_inr: float
    customer_total_past_orders: int
    customer_is_premium: int
    customer_complaints_90d: int
    cuisine_type: str
    weather: str
    traffic_level: str
    vehicle_type: str

class ChurnRequest(BaseModel):
    distance_km: float
    delivery_time_min: float
    customer_rating: int
    customer_total_past_orders: int
    customer_complaints_90d: int
    order_value_inr: float
    customer_is_premium: int

class RecommendRequest(BaseModel):
    user_id: str
    n: int = 5

class ContentRecommendRequest(BaseModel):
    food_name: str
    n: int = 5


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"message": "NourishNet AI API is running", "version": "1.0.0"}


@app.post("/predict/delivery-time")
def predict_delivery_time(req: DeliveryRequest):
    df = pd.DataFrame([req.dict()])
    prediction = delivery_model.predict(df)[0]
    return {
        "estimated_delivery_time_min": round(float(prediction), 1),
        "status": "success"
    }


@app.post("/predict/churn")
def predict_churn(req: ChurnRequest):
    df = pd.DataFrame([req.dict()])
    prediction = churn_model.predict(df)[0]
    probability = churn_model.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": "High" if probability > 0.6 else "Medium" if probability > 0.3 else "Low",
        "status": "success"
    }


@app.post("/recommend/collaborative")
def collaborative_recommend(req: RecommendRequest):
    if req.user_id not in user_item_matrix.index:
        return {"recommendations": [], "message": "User not found"}

    rated_items = set(
        user_item_matrix.loc[req.user_id][user_item_matrix.loc[req.user_id] > 0].index
    )
    sim_scores = user_similarity[req.user_id].drop(req.user_id).nlargest(10)
    scores = {}
    for sim_user, sim_score in zip(sim_scores.index, sim_scores.values):
        rated = user_item_matrix.loc[sim_user]
        for item, rating in rated.items():
            if rating > 0 and item not in rated_items:
                scores[item] = scores.get(item, 0) + sim_score * rating

    recommendations = sorted(scores, key=scores.get, reverse=True)[:req.n]
    return {"user_id": req.user_id, "recommendations": recommendations, "status": "success"}


@app.post("/recommend/content-based")
def content_recommend(req: ContentRecommendRequest):
    if req.food_name not in item_sim_df:
        return {"recommendations": [], "message": "Food item not found"}

    recs = item_sim_df[req.food_name].drop(req.food_name).nlargest(req.n).index.tolist()
    return {"food_name": req.food_name, "recommendations": recs, "status": "success"}


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}