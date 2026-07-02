from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.helpers import ensure_path_exists, load_pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NourishNet AI API",
    description="AI-powered food delivery intelligence platform",
    version="1.0.0",
)

MODEL_DIR = ROOT_DIR / "models" / "saved"

CV_CLASSES = ["donuts", "fried_rice", "ice_cream", "omelette", "pizza", "samosa", "sushi"]
SENTIMENT_LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

cv_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

FALLBACK_CONTENT_RECOMMENDATIONS = {
    "biryani": ["Paneer Tikka", "Butter Chicken", "Veg Fried Rice", "Dal Makhani", "Samosa"],
    "pizza": ["Margherita Pizza", "Sushi Roll", "Paneer Tikka", "Veg Fried Rice", "Ice Cream Cone"],
    "sushi": ["Paneer Tikka", "Margherita Pizza", "Chocolate Donut", "Veg Fried Rice", "Ice Cream Cone"],
    "burger": ["Paneer Tikka", "Veg Fried Rice", "Samosa", "Dal Makhani", "Margherita Pizza"],
    "pasta": ["Margherita Pizza", "Paneer Tikka", "Veg Fried Rice", "Samosa", "Chocolate Donut"],
    "default": ["Paneer Tikka", "Veg Fried Rice", "Margherita Pizza", "Samosa", "Dal Makhani"],
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": "Invalid input", "details": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unexpected error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


def initialize_app_state():
    app.state.delivery_model = None
    app.state.churn_model = None
    app.state.user_item_matrix = None
    app.state.user_similarity = None
    app.state.item_sim_df = None
    app.state.cv_model = None
    app.state.sentiment_tokenizer = None
    app.state.sentiment_model = None
    app.state.generator = None
    app.state.models_loaded = False
    app.state.model_errors = []


def load_pickled_model(filename: str):
    path = MODEL_DIR / filename
    return load_pickle(path)


def load_cv_model():
    path = MODEL_DIR / "best_food_classifier.pth"
    ensure_path_exists(path, "CV model weights")
    model = models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.last_channel, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, len(CV_CLASSES)),
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_sentiment_model():
    sentiment_dir = MODEL_DIR / "sentiment_model"
    ensure_path_exists(sentiment_dir, "Sentiment model directory")
    tokenizer = AutoTokenizer.from_pretrained(str(sentiment_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(sentiment_dir))
    model.eval()
    return tokenizer, model


def load_text_generator():
    try:
        return hf_pipeline("text-generation", model="facebook/opt-125m", max_new_tokens=80)
    except Exception:
        return None


def require_available(component: str, model):
    if model is None:
        raise HTTPException(status_code=503, detail=f"{component} is not available")
    return model


def estimate_delivery_time(req) -> float:
    weather_penalty = {"sunny": 0, "clear": 0, "cloudy": 4, "rain": 6, "storm": 14}.get(
        str(req.weather).lower(), 0
    )
    traffic_penalty = {"low": 0, "medium": 7, "high": 15}.get(
        str(req.traffic_level).lower(), 7
    )
    peak_penalty = 8 if bool(req.is_peak_hour) else 0
    complaints_penalty = 3 * int(req.customer_complaints_90d)
    premium_bonus = -2 if bool(req.customer_is_premium) else 0

    return round(
        10
        + 2.5 * float(req.distance_km)
        + 0.7 * float(req.restaurant_prep_time_min)
        + weather_penalty
        + traffic_penalty
        + peak_penalty
        + complaints_penalty
        + premium_bonus
        + (float(req.order_value_inr) / 1000.0),
        1,
    )


def estimate_churn_probability(req) -> float:
    risk_score = 0.15

    if float(req.delivery_time_min) > 45:
        risk_score += 0.2
    if int(req.customer_rating) <= 2:
        risk_score += 0.25
    if int(req.customer_complaints_90d) > 0:
        risk_score += 0.2
    if int(req.customer_total_past_orders) < 3:
        risk_score += 0.1
    if not bool(req.customer_is_premium):
        risk_score += 0.08
    if float(req.order_value_inr) < 200:
        risk_score += 0.07

    return max(0.05, min(0.95, risk_score))


def fallback_content_recommendations(food_name: str, n: int) -> list[str]:
    key = str(food_name).strip().lower()
    if key in FALLBACK_CONTENT_RECOMMENDATIONS:
        items = FALLBACK_CONTENT_RECOMMENDATIONS[key]
    else:
        items = FALLBACK_CONTENT_RECOMMENDATIONS["default"]

    return items[:n]


def _model_dump(payload):
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def load_models_into_state():
    initialize_app_state()
    errors = []

    try:
        app.state.delivery_model = load_pickled_model("delivery_time_best_classical.pkl")
    except Exception as exc:
        errors.append(str(exc))
        logger.warning("Delivery model failed to load: %s", exc)

    try:
        app.state.churn_model = load_pickled_model("churn_best_model.pkl")
    except Exception as exc:
        errors.append(str(exc))
        logger.warning("Churn model failed to load: %s", exc)

    try:
        app.state.user_item_matrix = load_pickled_model("recommendation_models/user_item_matrix.pkl")
        app.state.user_similarity = load_pickled_model("recommendation_models/user_similarity.pkl")
        app.state.item_sim_df = load_pickled_model("recommendation_models/item_similarity.pkl")
    except Exception as exc:
        errors.append(str(exc))
        logger.warning("Recommendation models failed to load: %s", exc)

    app.state.models_loaded = len(errors) == 0
    app.state.model_errors = errors


@app.on_event("startup")
def load_models():
    load_models_into_state()


initialize_app_state()


class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., ge=0)
    restaurant_prep_time_min: float = Field(..., ge=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    is_peak_hour: bool
    order_value_inr: float = Field(..., ge=0)
    customer_total_past_orders: int = Field(..., ge=0)
    customer_is_premium: bool
    customer_complaints_90d: int = Field(..., ge=0)
    cuisine_type: str
    weather: str
    traffic_level: str
    vehicle_type: str


class ChurnRequest(BaseModel):
    distance_km: float = Field(..., ge=0)
    delivery_time_min: float = Field(..., ge=0)
    customer_rating: int = Field(..., ge=1, le=5)
    customer_total_past_orders: int = Field(..., ge=0)
    customer_complaints_90d: int = Field(..., ge=0)
    order_value_inr: float = Field(..., ge=0)
    customer_is_premium: bool


class RecommendRequest(BaseModel):
    user_id: str
    n: int = Field(default=5, ge=1, le=20)


class ContentRecommendRequest(BaseModel):
    food_name: str
    n: int = Field(default=5, ge=1, le=20)


class SentimentRequest(BaseModel):
    review: str = Field(..., min_length=1)


class GenerateRequest(BaseModel):
    food_name: str = Field(..., min_length=1)


class RagRequest(BaseModel):
    query: str = Field(..., min_length=3)


@app.get("/")
def root():
    return {"message": "NourishNet AI API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": app.state.models_loaded,
        "component_status": {
            "delivery_model": app.state.delivery_model is not None,
            "churn_model": app.state.churn_model is not None,
            "recommendation_models": (
                app.state.user_item_matrix is not None
                and app.state.user_similarity is not None
                and app.state.item_sim_df is not None
            ),
            "cv_model": app.state.cv_model is not None,
            "sentiment_model": app.state.sentiment_model is not None,
            "text_generator": app.state.generator is not None,
        },
        "errors": app.state.model_errors,
    }


@app.post("/predict/delivery-time")
def predict_delivery_time(req: DeliveryRequest):
    if app.state.delivery_model is None:
        prediction = estimate_delivery_time(req)
        return {"estimated_delivery_time_min": prediction, "status": "success", "source": "fallback"}
    delivery_model = require_available("Delivery time model", app.state.delivery_model)
    df = pd.DataFrame([_model_dump(req)])
    prediction = delivery_model.predict(df)[0]
    return {"estimated_delivery_time_min": round(float(prediction), 1), "status": "success", "source": "model"}


@app.post("/predict/churn")
def predict_churn(req: ChurnRequest):
    if app.state.churn_model is None:
        probability = estimate_churn_probability(req)
        prediction = int(probability >= 0.5)
        return {
            "churn_prediction": prediction,
            "churn_probability": round(float(probability), 3),
            "risk_level": "High" if probability > 0.6 else "Medium" if probability > 0.3 else "Low",
            "status": "success",
            "source": "fallback",
        }

    churn_model = require_available("Churn model", app.state.churn_model)
    df = pd.DataFrame([_model_dump(req)])
    prediction = churn_model.predict(df)[0]
    probability = churn_model.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": "High" if probability > 0.6 else "Medium" if probability > 0.3 else "Low",
        "status": "success",
        "source": "model",
    }


@app.post("/predict/sentiment")
def predict_sentiment(req: SentimentRequest):
    tokenizer = require_available("Sentiment tokenizer", app.state.sentiment_tokenizer)
    sentiment_model = require_available("Sentiment model", app.state.sentiment_model)
    encoding = tokenizer(
        req.review,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = sentiment_model(**encoding)
        pred = outputs.logits.argmax(dim=1).item()
    return {"review": req.review, "sentiment": SENTIMENT_LABELS[pred], "status": "success"}


@app.post("/predict/food-image")
async def predict_food_image(file: UploadFile = File(...)):
    cv_model = require_available("CV model", app.state.cv_model)
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to parse uploaded image") from exc

    tensor = cv_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = cv_model(tensor)
        pred = outputs.argmax(dim=1).item()
    return {"predicted_class": CV_CLASSES[pred], "status": "success"}


@app.post("/generate/food-description")
def generate_description(req: GenerateRequest):
    if app.state.generator is None:
        app.state.generator = load_text_generator()

    generator = require_available("Text generator", app.state.generator)
    prompt = f"Write an appetizing menu description for {req.food_name}:"
    result = generator(prompt, do_sample=True, temperature=0.7)
    generated = result[0]["generated_text"].replace(prompt, "").strip()
    return {"food_name": req.food_name, "description": generated[:200], "status": "success"}


@app.post("/recommend/collaborative")
def collaborative_recommend(req: RecommendRequest):
    if app.state.user_item_matrix is None or app.state.user_similarity is None:
        return {
            "user_id": req.user_id,
            "recommendations": [
                "Paneer Tikka",
                "Veg Fried Rice",
                "Margherita Pizza",
                "Samosa",
                "Dal Makhani",
            ],
            "status": "success",
            "source": "fallback",
        }

    user_item_matrix = require_available("User item matrix", app.state.user_item_matrix)
    user_similarity = require_available("User similarity matrix", app.state.user_similarity)

    if req.user_id not in user_item_matrix.index or req.user_id not in user_similarity.index:
        raise HTTPException(status_code=404, detail="User not found")

    rated_items = set(user_item_matrix.loc[req.user_id][user_item_matrix.loc[req.user_id] > 0].index)
    sim_scores = user_similarity.loc[req.user_id].drop(labels=[req.user_id], errors="ignore").nlargest(10)

    scores = {}
    for sim_user, sim_score in sim_scores.items():
        rated = user_item_matrix.loc[sim_user]
        for item, rating in rated.items():
            if rating > 0 and item not in rated_items:
                scores[item] = scores.get(item, 0) + float(sim_score) * float(rating)

    recommendations = sorted(scores, key=scores.get, reverse=True)[:req.n]
    return {"user_id": req.user_id, "recommendations": recommendations, "status": "success"}


@app.post("/recommend/content-based")
def content_recommend(req: ContentRecommendRequest):
    if app.state.item_sim_df is None:
        recommendations = fallback_content_recommendations(req.food_name, req.n)
        return {
            "food_name": req.food_name,
            "recommendations": recommendations,
            "status": "success",
            "source": "fallback",
        }

    item_sim_df = require_available("Item similarity matrix", app.state.item_sim_df)
    if req.food_name not in item_sim_df.index:
        recommendations = fallback_content_recommendations(req.food_name, req.n)
        return {
            "food_name": req.food_name,
            "recommendations": recommendations,
            "status": "success",
            "source": "fallback",
        }

    recs = item_sim_df.loc[req.food_name].drop(labels=[req.food_name], errors="ignore").nlargest(req.n).index.tolist()
    return {
        "food_name": req.food_name,
        "recommendations": recs,
        "status": "success",
        "source": "model",
    }


@app.post("/predict/rag")
def predict_rag(req: RagRequest):
    raise HTTPException(
        status_code=501,
        detail="RAG support is not configured in this deployment. Add retrieval model artifacts and a RAG pipeline to enable this endpoint.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)