"""
NourishNet AI - FastAPI Service
Serves delivery time prediction, churn prediction, sentiment analysis,
food image classification, generative AI, and food recommendations.
"""

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NourishNet AI API",
    description="AI-powered food delivery intelligence platform",
    version="1.0.0"
)

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": "Invalid input", "details": str(exc)})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

MODEL_DIR = Path("models/saved")

# Load models
delivery_model  = joblib.load(MODEL_DIR / "delivery_time_best_classical.pkl")
churn_model     = joblib.load(MODEL_DIR / "churn_best_model.pkl")
user_item_matrix = joblib.load(MODEL_DIR / "recommendation_models/user_item_matrix.pkl")
user_similarity  = joblib.load(MODEL_DIR / "recommendation_models/user_similarity.pkl")
item_sim_df      = joblib.load(MODEL_DIR / "recommendation_models/item_similarity.pkl")

# Load CV model
import torch
import torchvision.transforms as transforms
import torchvision.models as models

CV_CLASSES = ['donuts', 'fried_rice', 'ice_cream', 'omelette', 'pizza', 'samosa', 'sushi']

def load_cv_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(m.last_channel, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, len(CV_CLASSES))
    )
    m.load_state_dict(torch.load(MODEL_DIR / "best_food_classifier.pth", map_location="cpu"))
    m.eval()
    return m

cv_model = load_cv_model()
cv_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Sentiment model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
SENTIMENT_LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
sentiment_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "sentiment_model"))
sentiment_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR / "sentiment_model"))
sentiment_model.eval()

# Load Generative AI model
from transformers import pipeline as hf_pipeline
generator = hf_pipeline("text-generation", model="facebook/opt-125m", max_new_tokens=80)


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

class SentimentRequest(BaseModel):
    review: str

class GenerateRequest(BaseModel):
    food_name: str


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"message": "NourishNet AI API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict/delivery-time")
def predict_delivery_time(req: DeliveryRequest):
    df = pd.DataFrame([req.dict()])
    prediction = delivery_model.predict(df)[0]
    logger.info(f"Delivery prediction: {round(float(prediction), 1)} min")
    return {"estimated_delivery_time_min": round(float(prediction), 1), "status": "success"}


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


@app.post("/predict/sentiment")
def predict_sentiment(req: SentimentRequest):
    encoding = sentiment_tokenizer(
        req.review, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = sentiment_model(**encoding)
        pred = outputs.logits.argmax(dim=1).item()
    return {"review": req.review, "sentiment": SENTIMENT_LABELS[pred], "status": "success"}


@app.post("/predict/food-image")
async def predict_food_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = cv_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = cv_model(tensor)
        pred = outputs.argmax(dim=1).item()
    return {"predicted_class": CV_CLASSES[pred], "status": "success"}


@app.post("/generate/food-description")
def generate_description(req: GenerateRequest):
    prompt = f"Write an appetizing menu description for {req.food_name}:"
    result = generator(prompt, do_sample=True, temperature=0.7)
    generated = result[0]["generated_text"].replace(prompt, "").strip()
    return {"food_name": req.food_name, "description": generated[:200], "status": "success"}


@app.post("/recommend/collaborative")
def collaborative_recommend(req: RecommendRequest):
    if req.user_id not in user_item_matrix.index:
        return {"recommendations": [], "message": "User not found"}
    rated_items = set(user_item_matrix.loc[req.user_id][user_item_matrix.loc[req.user_id] > 0].index)
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