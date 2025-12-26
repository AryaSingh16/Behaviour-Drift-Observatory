from pydantic import BaseModel
from typing import List


class DriftPoint(BaseModel):
    day: int
    drift_score: float


class DriftTimeline(BaseModel):
    user_id: str
    timeline: List[DriftPoint]


class DriftExplanation(BaseModel):
    feature: str
    contribution: float
    direction: str


class DriftExplanationResponse(BaseModel):
    user_id: str
    day: int
    explanations: List[DriftExplanation]
