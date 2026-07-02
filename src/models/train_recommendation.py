"""
NourishNet AI - Recommendation System
Collaborative Filtering + Content-Based Filtering
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = Path("models/saved/recommendation_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FOOD_ITEMS = [
    "Butter Chicken", "Masala Dosa", "Biryani", "Pizza Margherita",
    "Veg Fried Rice", "Paneer Tikka", "Sushi Platter", "Chocolate Ice Cream",
    "Samosa", "Pasta Arrabiata", "Garlic Naan", "Mango Lassi",
    "Chicken Burger", "Dal Makhani", "Gulab Jamun"
]

FOOD_METADATA = {
    "Butter Chicken": "chicken curry creamy indian spicy",
    "Masala Dosa": "south indian crispy crepe potato vegetarian",
    "Biryani": "rice chicken spicy indian aromatic",
    "Pizza Margherita": "italian pizza cheese tomato vegetarian",
    "Veg Fried Rice": "chinese rice vegetarian stir-fry",
    "Paneer Tikka": "indian vegetarian paneer grilled spicy",
    "Sushi Platter": "japanese seafood rice fresh healthy",
    "Chocolate Ice Cream": "dessert sweet cold chocolate creamy",
    "Samosa": "indian snack fried potato crispy vegetarian",
    "Pasta Arrabiata": "italian pasta spicy tomato vegetarian",
    "Garlic Naan": "indian bread garlic butter baked",
    "Mango Lassi": "indian drink mango yogurt sweet",
    "Chicken Burger": "fast food chicken grilled burger bun",
    "Dal Makhani": "indian lentil creamy vegetarian curry",
    "Gulab Jamun": "indian dessert sweet fried milk syrup",
}

def generate_interactions(n_users=300):
    np.random.seed(42)
    N_ITEMS = len(FOOD_ITEMS)
    ratings = []
    for user_id in range(n_users):
        n_rated = np.random.randint(3, 10)
        items_rated = np.random.choice(N_ITEMS, size=n_rated, replace=False)
        for item_id in items_rated:
            rating = np.random.choice([3, 4, 4, 5, 5], p=[0.1, 0.2, 0.3, 0.2, 0.2])
            ratings.append({"user_id": f"U{user_id:04d}", "item_id": item_id,
                            "food_name": FOOD_ITEMS[item_id], "rating": rating})
    return pd.DataFrame(ratings)

if __name__ == "__main__":
    print("Generating interaction data...")
    df = generate_interactions()

    # Collaborative filtering
    print("Building collaborative filtering model...")
    user_item_matrix = df.pivot_table(index="user_id", columns="food_name", values="rating").fillna(0)
    user_similarity  = pd.DataFrame(
        cosine_similarity(user_item_matrix),
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    # Content-based filtering
    print("Building content-based filtering model...")
    items = list(FOOD_METADATA.keys())
    descriptions = list(FOOD_METADATA.values())
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(descriptions)
    item_similarity = pd.DataFrame(cosine_similarity(tfidf_matrix), index=items, columns=items)

    # Save all models
    joblib.dump(user_item_matrix, MODEL_DIR / "user_item_matrix.pkl")
    joblib.dump(user_similarity,  MODEL_DIR / "user_similarity.pkl")
    joblib.dump(item_similarity,  MODEL_DIR / "item_similarity.pkl")
    joblib.dump(tfidf,            MODEL_DIR / "tfidf_vectorizer.pkl")

    print("All recommendation models saved.")

    # Test
    test_user = "U0001"
    rated = df[df["user_id"] == test_user]["food_name"].tolist()
    print(f"\nUser {test_user} ordered: {rated}")

    rated_set = set(user_item_matrix.loc[test_user][user_item_matrix.loc[test_user] > 0].index)
    sim_scores = user_similarity[test_user].drop(test_user).nlargest(10)
    scores = {}
    for sim_user, sim_score in zip(sim_scores.index, sim_scores.values):
        for item, rating in user_item_matrix.loc[sim_user].items():
            if rating > 0 and item not in rated_set:
                scores[item] = scores.get(item, 0) + sim_score * rating
    recs = sorted(scores, key=scores.get, reverse=True)[:5]
    print(f"Recommendations: {recs}")