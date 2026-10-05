
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "movie_revenue_pipeline.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Movie Investment Prediction API"
)


class MovieInput(BaseModel):
    budget_clean: float = Field(..., gt=0)
    popularity: float = Field(..., ge=0)
    runtime_clean: float = Field(..., gt=0)
    vote_average: float = Field(..., ge=0, le=10)
    vote_count: float = Field(..., ge=0)
    cast_size: float = Field(..., ge=0)
    crew_size: float = Field(..., ge=0)
    keyword_count: float = Field(..., ge=0)
    genre_count: float = Field(..., ge=0)
    release_year: float
    release_month: float = Field(..., ge=1, le=12)
    full_rating_mean: float = Field(..., ge=0, le=5)
    full_rating_count: float = Field(..., ge=0)
    primary_genre: str
    is_english: bool


def investment_category(roi):
    if roi >= 50:
        return "High Potential"
    elif roi >= 0:
        return "Moderate Potential"
    else:
        return "Low Potential"


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Movie Investment Prediction API is running"
    }


@app.post("/predict")
def predict_movie(movie: MovieInput):
    try:
        input_data = pd.DataFrame([{
            "budget_clean": movie.budget_clean,
            "popularity": movie.popularity,
            "runtime_clean": movie.runtime_clean,
            "vote_average": movie.vote_average,
            "vote_count": movie.vote_count,
            "cast_size": movie.cast_size,
            "crew_size": movie.crew_size,
            "keyword_count": movie.keyword_count,
            "genre_count": movie.genre_count,
            "release_year": movie.release_year,
            "release_month": movie.release_month,
            "full_rating_mean": movie.full_rating_mean,
            "full_rating_count": movie.full_rating_count,
            "primary_genre": movie.primary_genre,
            "is_english": movie.is_english
        }])

        predicted_revenue = float(model.predict(input_data)[0])
        predicted_revenue = max(predicted_revenue, 0)

        predicted_profit = (
            predicted_revenue - movie.budget_clean
        )

        predicted_roi = (
            predicted_profit / movie.budget_clean
        ) * 100

        potential = investment_category(
            predicted_roi
        )

        return {
            "predicted_revenue": round(predicted_revenue, 2),
            "predicted_profit": round(predicted_profit, 2),
            "predicted_roi_percent": round(predicted_roi, 2),
            "investment_potential": potential
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
