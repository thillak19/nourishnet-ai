from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.helpers import ensure_dir, save_pickle

MODEL_DIR = ROOT_DIR / "models" / "saved" / "recommendation_models"
ensure_dir(MODEL_DIR)

FOOD_ITEMS = [
    "Butter Chicken",
    "Masala Dosa",
    "Biryani",
    "Pizza Margherita",
    "Veg Fried Rice",
    "Paneer Tikka",
    "Sushi Platter",
    "Chocolate Ice Cream",
    "Samosa",
    "Pasta Arrabiata",
    "Garlic Naan",
    "Mango Lassi",
    "Chicken Burger",
    "Dal Makhani",
    "Gulab Jamun",
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


def generate_interactions(n_users: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    n_items = len(FOOD_ITEMS)
    ratings = []

    for user_id in range(n_users):
        n_rated = np.random.randint(3, 10)
        items_rated = np.random.choice(n_items, size=n_rated, replace=False)
        for item_id in items_rated:
            rating = np.random.choice([3, 4, 4, 5, 5], p=[0.1, 0.2, 0.3, 0.2, 0.2])
            ratings.append(
                {
                    "user_id": f"U{user_id:04d}",
                    "item_id": item_id,
                    "food_name": FOOD_ITEMS[item_id],
                    "rating": rating,
                }
            )

    return pd.DataFrame(ratings)


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(index="user_id", columns="food_name", values="rating", aggfunc="mean")
        .fillna(0)
        .astype(float)
    )


def build_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        cosine_similarity(user_item_matrix),
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )


def build_item_similarity(items: list[str], metadata: dict[str, str]):
    descriptions = [metadata[item] for item in items]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(descriptions)
    similarity = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(similarity, index=items, columns=items), tfidf


def save_recommendation_models(
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    item_similarity: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
) -> None:
    save_pickle(user_item_matrix, MODEL_DIR / "user_item_matrix.pkl")
    save_pickle(user_similarity, MODEL_DIR / "user_similarity.pkl")
    save_pickle(item_similarity, MODEL_DIR / "item_similarity.pkl")
    save_pickle(tfidf_vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")


def print_test_recommendations(
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
) -> None:
    test_user = "U0001"
    if test_user not in user_item_matrix.index:
        return

    rated_set = set(
        user_item_matrix.loc[test_user][user_item_matrix.loc[test_user] > 0].index
    )
    sim_scores = user_similarity[test_user].drop(test_user).nlargest(10)
    scores: dict[str, float] = {}

    for sim_user, sim_score in sim_scores.items():
        for item, rating in user_item_matrix.loc[sim_user].items():
            if rating > 0 and item not in rated_set:
                scores[item] = scores.get(item, 0.0) + sim_score * rating

    recs = sorted(scores, key=scores.get, reverse=True)[:5]
    print(f"User {test_user} ordered: {sorted(rated_set)}")
    print(f"Recommendations: {recs}")


def main() -> None:
    print("Generating interaction data...")
    df = generate_interactions()

    print("Building collaborative filtering model...")
    user_item_matrix = build_user_item_matrix(df)
    user_similarity = build_user_similarity(user_item_matrix)

    print("Building content-based filtering model...")
    item_similarity, tfidf = build_item_similarity(FOOD_ITEMS, FOOD_METADATA)

    save_recommendation_models(user_item_matrix, user_similarity, item_similarity, tfidf)
    print("All recommendation models saved.")

    print_test_recommendations(user_item_matrix, user_similarity)


if __name__ == "__main__":
    main()