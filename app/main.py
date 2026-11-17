"""FastAPI service: raw job posting in -> fraud risk score + verdict out.

Run:   uvicorn app.main:app --reload        (from the project root)
Docs:  http://127.0.0.1:8000/docs           (interactive Swagger UI with ready-made examples)
Try:   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @app/examples/fake_posting.json
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.preprocess import INPUT_FIELDS, ROOT  # the pickled pipeline also needs src/ importable

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "fraud_detector.joblib"))
EXAMPLES = {name: json.loads((Path(__file__).parent / "examples" / f"{name}.json").read_text())
            for name in ["fake_posting", "real_posting"]}
MODEL: dict = {}  # filled once at startup

ShortText = Annotated[str | None, Field(max_length=300)]
LongText = Annotated[str | None, Field(max_length=20_000)]  # longest field in the dataset is ~15k chars


class JobPosting(BaseModel):
    """A job posting, using the same field names as the training data."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)  # typo'd field names -> 422, not ignored

    title: str = Field(min_length=1, max_length=300)
    company_profile: LongText = None
    description: str = Field(min_length=1, max_length=20_000)
    requirements: LongText = None
    benefits: LongText = None
    # Fixed category lists from the dataset, so "full time" is rejected instead of silently becoming "unknown"
    employment_type: Literal["Full-time", "Part-time", "Contract", "Temporary", "Other"] | None = None
    required_experience: Literal["Internship", "Entry level", "Associate", "Mid-Senior level", "Director",
                                 "Executive", "Not Applicable"] | None = None
    required_education: Literal["High School or equivalent", "Some High School Coursework", "Some College Coursework Completed",
                                "Associate Degree", "Bachelor's Degree", "Master's Degree", "Doctorate", "Professional",
                                "Certification", "Vocational", "Vocational - Degree", "Vocational - HS Diploma",
                                "Unspecified"] | None = None
    industry: ShortText = None   # free text: categories unseen in training fall into the "rare" bucket
    function: ShortText = None
    location: ShortText = Field(None, examples=["US, NY, New York"])
    department: ShortText = None
    salary_range: ShortText = Field(None, examples=["40000-50000"])
    # Required: they are never missing in the training data, and a silent default would bias the score
    telecommuting: bool
    has_company_logo: bool
    has_questions: bool = Field(description="Does the application include screening questions?")


assert set(JobPosting.model_fields) == set(INPUT_FIELDS), "API schema out of sync with the model's input fields"


class Prediction(BaseModel):
    fraud_probability: float = Field(description="Model risk score in [0, 1]. Not calibrated (SMOTE training inflates "
                                                 "it): compare it to the threshold, don't read it as a true probability.")
    is_fraud: bool = Field(description="fraud_probability >= threshold")
    threshold: float = Field(description="Tuned on out-of-fold training predictions for precision >= 0.80")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup (not per request); fail fast if it's missing."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No model at {MODEL_PATH}. Train it first: python -m src.train")
    bundle = joblib.load(MODEL_PATH)
    bundle["pipeline"].named_steps["model"].set_params(n_jobs=1)  # one row at a time: threads cost more than they save
    MODEL.update(bundle)
    yield
    MODEL.clear()


app = FastAPI(title="Fake Job Posting Detector", version="1.0", lifespan=lifespan,
              description="Scores a job posting for fraud risk with a Random Forest trained on the EMSCAD dataset.")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL["model"], "imbalance": MODEL["imbalance"],
            "threshold": round(MODEL["threshold"], 4), "trained_on": MODEL["trained_on"]}


@app.post("/predict", response_model=Prediction)
def predict(posting: Annotated[JobPosting, Body(openapi_examples={
        "fake": {"summary": "Likely fake: work-from-home data entry", "value": EXAMPLES["fake_posting"]},
        "real": {"summary": "Likely real: backend engineer", "value": EXAMPLES["real_posting"]}})]):
    # Plain `def`, not `async def`: FastAPI runs it in a worker thread, so CPU-bound scoring can't block the server
    X = pd.DataFrame([posting.model_dump()])
    proba = float(MODEL["pipeline"].predict_proba(X)[0, 1])
    t = MODEL["threshold"]
    return Prediction(fraud_probability=round(proba, 4), is_fraud=proba >= t, threshold=round(t, 4))
