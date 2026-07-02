from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

CUISINES = ["Indian", "Chinese", "Italian", "Fast Food", "South Indian", "Continental"]
WEATHER = ["Sunny", "Cloudy", "Rain", "Storm"]
TRAFFIC = ["Low", "Medium", "High"]
VEHICLES = ["Bike", "Scooter", "Bicycle"]
FOOD_ITEMS = [
    "Paneer Tikka",
    "Margherita Pizza",
    "Sushi Roll",
    "Chocolate Donut",
    "Veg Fried Rice",
    "Ice Cream Cone",
]
USER_IDS = [f"user_{i:03d}" for i in range(1, 101)]


def generate_order_row(order_id: int) -> dict:
    cuisine = random.choice(CUISINES)
    weather = random.choice(WEATHER)
    traffic_level = random.choice(TRAFFIC)
    vehicle_type = random.choice(VEHICLES)
    is_peak_hour = int(random.random() < 0.35)
    distance_km = round(random.uniform(0.5, 12.0), 2)
    restaurant_prep_time_min = random.randint(5, 30)
    order_value_inr = round(random.uniform(100, 1500), 2)
    customer_total_past_orders = random.randint(0, 75)
    customer_is_premium = int(random.random() < 0.25)
    customer_complaints_90d = random.randint(0, 3)

    weather_penalty = {"Sunny": 0, "Cloudy": 4, "Rain": 6, "Storm": 14}[weather]
    traffic_penalty = {"Low": 0, "Medium": 7, "High": 15}[traffic_level]
    peak_penalty = 8 if is_peak_hour else 0

    delivery_time_min = round(
        10
        + 2.5 * distance_km
        + 0.7 * restaurant_prep_time_min
        + weather_penalty
        + traffic_penalty
        + peak_penalty
        + random.uniform(-3, 3),
        1,
    )

    customer_rating = random.randint(1, 5)
    churned = int(random.random() < 0.12 + 0.05 * (customer_complaints_90d > 0))

    return {
        "order_id": order_id,
        "user_id": random.choice(USER_IDS),
        "food_name": random.choice(FOOD_ITEMS),
        "distance_km": distance_km,
        "restaurant_prep_time_min": restaurant_prep_time_min,
        "hour_of_day": random.randint(8, 23),
        "is_peak_hour": is_peak_hour,
        "order_value_inr": order_value_inr,
        "customer_total_past_orders": customer_total_past_orders,
        "customer_is_premium": customer_is_premium,
        "customer_complaints_90d": customer_complaints_90d,
        "cuisine_type": cuisine,
        "weather": weather,
        "traffic_level": traffic_level,
        "vehicle_type": vehicle_type,
        "delivery_time_min": delivery_time_min,
        "customer_rating": customer_rating,
        "churned": churned,
        "rating": random.randint(1, 5),
    }


def generate_dataset(n: int = 2000, output_file: Path | str | None = None) -> Path:
    output_path = Path(output_file) if output_file is not None else RAW_DIR / "orders.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "order_id",
        "user_id",
        "food_name",
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
        "delivery_time_min",
        "customer_rating",
        "churned",
        "rating",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for order_id in range(1, n + 1):
            writer.writerow(generate_order_row(order_id))

    return output_path


if __name__ == "__main__":
    output_path = generate_dataset(2000)
    print(f"Generated dataset at: {output_path}")