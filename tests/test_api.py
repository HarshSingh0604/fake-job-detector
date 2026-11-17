"""API tests. Run from the project root: python -m pytest -q   (needs a trained model in models/)"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import EXAMPLES, MODEL, MODEL_PATH, app

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="no model yet: run python -m src.train")
MINIMAL = {"title": "Clerk", "description": "Office work.", "telecommuting": False,
           "has_company_logo": True, "has_questions": False}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the startup event, i.e. loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_response_format(client):
    body = client.post("/predict", json=MINIMAL).json()
    assert set(body) == {"fraud_probability", "is_fraud", "threshold"}
    assert 0 <= body["fraud_probability"] <= 1


def test_example_verdicts(client):
    assert client.post("/predict", json=EXAMPLES["fake_posting"]).json()["is_fraud"] is True
    assert client.post("/predict", json=EXAMPLES["real_posting"]).json()["is_fraud"] is False


def test_api_matches_offline_pipeline(client):
    """Train/serve consistency: the API must return the same score as calling the saved pipeline directly."""
    for posting in EXAMPLES.values():
        offline = MODEL["pipeline"].predict_proba(pd.DataFrame([posting]))[0, 1]
        assert client.post("/predict", json=posting).json()["fraud_probability"] == pytest.approx(offline, abs=1e-4)


@pytest.mark.parametrize("bad", [
    {k: v for k, v in MINIMAL.items() if k != "has_company_logo"},  # required field missing
    {**MINIMAL, "title": "   "},                                      # blank after stripping
    {**MINIMAL, "employment_type": "full time"},                      # not one of the allowed categories
    {**MINIMAL, "descripton": "typo"},                                # unknown field name
    {**MINIMAL, "has_questions": "maybe"},                            # not a boolean
    {**MINIMAL, "description": "x" * 20_001},                         # too long
])
def test_invalid_input_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422
