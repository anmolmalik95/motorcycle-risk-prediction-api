from pydantic import BaseModel, Field
from typing import Literal

class RiskRequest(BaseModel):
    temperature: float = Field(..., gt=-20, lt=60, description="Temperature in °C (-20 to 60)")
    rainfall: float = Field(..., ge=0, le=200, description="Rainfall in mm (0 to 200)")
    visibility: float = Field(..., ge=0, le=50, description="Visibility in km (0 to 50)")
    distance: float = Field(..., ge=0, le=2000, description="Planned ride distance in km (0 to 2000)")
    time_of_day: Literal["morning", "afternoon", "evening", "night"] = Field(
        ..., description="Time of day (fixed options)")
    experience: int = Field(..., ge=0, le=50, description="Rider experience in years (0 to 50)")

class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    advice: str