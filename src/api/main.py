from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path

from .schemas import (
    DriftPoint,
    DriftTimeline,
    DriftExplanation,
    DriftExplanationResponse,
)

# -------------------------
# App init
# -------------------------
app = FastAPI(
    title="Behavior Drift Observatory",
    description="Unsupervised behavioral drift monitoring system",
    version="1.0",
)

# -------------------------
# Load data once (startup)
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DRIFT_PATH = BASE_DIR / "data" / "drift_scores.csv"
EXPLAIN_PATH = BASE_DIR / "data" / "drift_explanations.csv"

drift_df = pd.read_csv(DRIFT_PATH)
explain_df = pd.read_csv(EXPLAIN_PATH)

# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Full drift timeline
# -------------------------
@app.get("/drift/score/{user_id}", response_model=DriftTimeline)
def get_drift_timeline(user_id: str):
    df = drift_df[drift_df["user_id"] == user_id]

    if df.empty:
        raise HTTPException(status_code=404, detail="User not found")

    timeline = [
        DriftPoint(day=int(row.day), drift_score=float(row.drift_score))
        for row in df.itertuples()
    ]

    return DriftTimeline(user_id=user_id, timeline=timeline)

# -------------------------
# Latest drift score
# -------------------------
@app.get("/drift/latest/{user_id}")
def get_latest_drift(user_id: str):
    df = drift_df[drift_df["user_id"] == user_id]

    if df.empty:
        raise HTTPException(status_code=404, detail="User not found")

    latest = df.sort_values("day").iloc[-1]

    return {
        "user_id": user_id,
        "day": int(latest.day),
        "drift_score": float(latest.drift_score),
    }

# -------------------------
# Drift explanation
# -------------------------
@app.get("/drift/explanation/{user_id}", response_model=DriftExplanationResponse)
def get_drift_explanation(user_id: str):
    user_explain = explain_df[explain_df["user_id"] == user_id]

    if user_explain.empty:
        raise HTTPException(status_code=404, detail="User not found")

    # Pick day with strongest total drift
    strongest_day = (
        user_explain.groupby("day")["contribution"]
        .apply(lambda x: x.abs().sum())
        .idxmax()
    )

    day_df = user_explain[user_explain["day"] == strongest_day]

    explanations = [
        DriftExplanation(
            feature=row.feature,
            contribution=float(row.contribution),
            direction=row.direction,
        )
        for row in day_df.itertuples()
    ]

    return DriftExplanationResponse(
        user_id=user_id,
        day=int(strongest_day),
        explanations=explanations,
    )
