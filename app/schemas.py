from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Study(BaseModel):
    model_config = ConfigDict(extra="ignore")

    study_id: str
    study_description: str = ""
    study_date: str | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: Study
    prior_studies: list[Study] = Field(default_factory=list)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    challenge_id: str | None = None
    schema_version: int | None = None
    generated_at: str | None = None
    cases: list[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]
