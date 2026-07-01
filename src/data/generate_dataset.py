"""
NourishNet AI - Synthetic Food Delivery Dataset Generator
Generates a realistic dataset for delivery time prediction and customer churn modeling.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random

np.random.seed(42)
random.seed(42)
fake = Faker()

N_ORDERS = 8000
N_CUSTOMERS = 1500

CUISINES = ["Indian", "Chinese", "Italian", "Fast Food", "South Indian", "Continental", "Desserts", "Beverages"]
WEATHER = ["Clear", "Rain", "Storm", "Cloudy"]
TRAFFIC = ["Low", "Medium", "High", "Jam"]
VEHICLE = ["Bike", "Scooter", "Bicycle"]


def generate_customers(n):
    customers = []
    for i in range(n):
        signup_orders = np.random.poisson(12) + 1
        avg_rating_given = round(np.random.normal(4.2, 0.6), 1)
        avg_rating_given = min(5.0, max(1.0, avg_rating_given))
        customers.append({
            "customer_id": f"CUST{i:05d}",
            "total_past_orders": signup_orders,
            "avg_rating_given": avg_rating_given,
            "is_premium_member": np.random.choice([0, 1], p=[0.7, 0.3]),
            "complaints_last_90d": np.random.poisson(0.3),
        })
    return pd.DataFrame(customers)


def generate_orders(n, customers_df):
    orders = []
    for i in range(n):
        cust = customers_df.sample(1).iloc[0]
        distance_km = round(np.random.gamma(2.5, 1.4), 2)
        distance_km = min(distance_km, 20.0)
        weather = np.random.choice(WEATHER, p=[0.55, 0.25, 0.05, 0.15])
        traffic = np.random.choice(TRAFFIC, p=[0.3, 0.35, 0.25, 0.10])
        vehicle = np.random.choice(VEHICLE, p=[0.6, 0.3, 0.1])
        prep_time = round(np.random.gamma(3, 4), 1)
        hour_of_day = np.random.choice(range(24), p=_hourly_demand_weights())
        is_peak = 1 if hour_of_day in [12, 13, 19, 20, 21] else 0

        base_time = 8 + distance_km * 2.8
        weather_penalty = {"Clear": 0, "Cloudy": 1, "Rain": 6, "Storm": 14}[weather]
        traffic_penalty = {"Low": 0, "Medium": 4, "High": 9, "Jam": 18}[traffic]
        vehicle_factor = {"Bike": 1.0, "Scooter": 1.05, "Bicycle": 1.6}[vehicle]
        peak_penalty = 6 if is_peak else 0

        delivery_time = (base_time + weather_penalty + traffic_penalty + peak_penalty) * vehicle_factor
        delivery_time += prep_time * 0.4
        delivery_time += np.random.normal(0, 3.5)
        delivery_time = max(10, round(delivery_time, 1))

        rating = 5 - (delivery_time > 45) * np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        rating = max(1, min(5, rating + np.random.choice([0, 0, 1, -1])))

        churn_score = (
            (delivery_time > 50) * 0.3
            + (rating <= 2) * 0.35
            + (cust["complaints_last_90d"] > 1) * 0.25
            + (cust["total_past_orders"] < 3) * 0.2
            - (cust["is_premium_member"]) * 0.15
            + np.random.normal(0, 0.15)
        )
        churned = 1 if churn_score > 0.22 else 0

        orders.append({
            "order_id": f"ORD{i:06d}",
            "customer_id": cust["customer_id"],
            "cuisine_type": np.random.choice(CUISINES),
            "distance_km": distance_km,
            "weather": weather,
            "traffic_level": traffic,
            "vehicle_type": vehicle,
            "restaurant_prep_time_min": prep_time,
            "hour_of_day": hour_of_day,
            "is_peak_hour": is_peak,
            "order_value_inr": round(np.random.gamma(3, 110), 2),
            "delivery_time_min": delivery_time,
            "customer_rating": rating,
            "customer_total_past_orders": cust["total_past_orders"],
            "customer_is_premium": cust["is_premium_member"],
            "customer_complaints_90d": cust["complaints_last_90d"],
            "churned": churned,
        })
    return pd.DataFrame(orders)


def _hourly_demand_weights():
    w = np.ones(24)
    for h in [12, 13, 19, 20, 21]:
        w[h] = 4
    for h in [7, 8, 9, 23, 0, 1, 2, 3, 4, 5]:
        w[h] = 0.3
    return w / w.sum()


if __name__ == "__main__":
    print("Generating customers...")
    customers_df = generate_customers(N_CUSTOMERS)

    print("Generating orders...")
    orders_df = generate_orders(N_ORDERS, customers_df)

    customers_df.to_csv("data/raw/customers.csv", index=False)
    orders_df.to_csv("data/raw/orders.csv", index=False)

    print(f"Done. customers.csv: {len(customers_df)} rows | orders.csv: {len(orders_df)} rows")
    print("\nOrders sample:")
    print(orders_df.head())
    print("\nChurn rate:", round(orders_df['churned'].mean(), 3))