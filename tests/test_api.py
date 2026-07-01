"""
NourishNet AI - API Tests
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "NourishNet AI API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_delivery_time_prediction():
    payload = {
        "distance_km": 5.0,
        "restaurant_prep_time_min": 15.0,
        "hour_of_day": 12,
        "is_peak_hour": 1,
        "order_value_inr": 350.0,
        "customer_total_past_orders": 10,
        "customer_is_premium": 0,
        "customer_complaints_90d": 0,
        "cuisine_type": "Indian",
        "weather": "Clear",
        "traffic_level": "Medium",
        "vehicle_type": "Bike"
    }
    response = client.post("/predict/delivery-time", json=payload)
    assert response.status_code == 200
    assert "estimated_delivery_time_min" in response.json()


def test_churn_prediction():
    payload = {
        "distance_km": 5.0,
        "delivery_time_min": 55.0,
        "customer_rating": 2,
        "customer_total_past_orders": 2,
        "customer_complaints_90d": 2,
        "order_value_inr": 200.0,
        "customer_is_premium": 0
    }
    response = client.post("/predict/churn", json=payload)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "risk_level" in response.json()


def test_content_recommendation():
    payload = {
        "food_name": "Biryani",
        "n": 5
    }
    response = client.post("/recommend/content-based", json=payload)
    assert response.status_code == 200
    assert "recommendations" in response.json()