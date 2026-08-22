"""
schemas.py
----------
Pydantic models describing the FastAPI request/response contracts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted defect category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence of the predicted class")
    inspection_status: Literal["ACCEPT", "REJECT", "MANUAL_REVIEW"] = Field(
        ..., description="Automated line decision derived from the prediction"
    )
    class_probabilities: dict[str, float] = Field(
        ..., description="Full softmax distribution over all defect classes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "pitted_surface",
                "confidence": 0.94,
                "inspection_status": "REJECT",
                "class_probabilities": {
                    "crazing": 0.01,
                    "inclusion": 0.02,
                    "patches": 0.01,
                    "pitted_surface": 0.94,
                    "rolled-in_scale": 0.01,
                    "scratches": 0.01,
                },
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]


class ErrorResponse(BaseModel):
    detail: str
