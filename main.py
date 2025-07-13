from fastapi import FastAPI
from typing import List
import pandas as pd

from recommender import recommend_hybrid_for_user  # import your function

# Load data once at startup
brands = pd.read_csv("brands.csv")
users = pd.read_csv("users.csv")
ratings = pd.read_csv("user_brand_ratings.csv")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Brand Recommendation API is running."}

@app.get("/recommend/{user_id}")
def get_recommendations(
    user_id: str,
    n: int = 5,
    k: int = 9,
    alpha: float = 0.7,
    min_similarity: float = 0.05
):
    results = recommend_hybrid_for_user(
        user_id=user_id,
        k=k,
        n_recommendations=n,
        alpha=alpha,
        min_similarity=min_similarity
    )
    return {"user_id": user_id, "recommendations": results}

