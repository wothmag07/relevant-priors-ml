"""Contract test: feed the example request from the challenge brief through the
endpoint and verify the response shape exactly matches the spec."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

EXAMPLE = {
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "generated_at": "2026-04-16T12:00:00.000Z",
    "cases": [
        {
            "case_id": "1001016",
            "patient_id": "606707",
            "patient_name": "Andrews, Micheal",
            "current_study": {
                "study_id": "3100042",
                "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                "study_date": "2026-03-08",
            },
            "prior_studies": [
                {"study_id": "2453245", "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST", "study_date": "2020-03-08"},
                {"study_id": "992654", "study_description": "CT HEAD WITHOUT CNTRST", "study_date": "2021-03-08"},
                {"study_id": "777777", "study_description": "KNEE, LEFT - 1 OR 2 VIEWS", "study_date": "2019-01-01"},
            ],
        }
    ],
}


def test_predict_contract_shape():
    client = TestClient(app)
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"predictions"}
    preds = body["predictions"]
    assert len(preds) == 3
    for pred in preds:
        assert set(pred.keys()) == {"case_id", "study_id", "predicted_is_relevant"}
        assert pred["case_id"] == "1001016"
        assert isinstance(pred["predicted_is_relevant"], bool)
    by_id = {p["study_id"]: p["predicted_is_relevant"] for p in preds}
    # Same MRI brain prior should be relevant.
    assert by_id["2453245"] is True
    # CT head prior should be relevant (same region, cross-modality).
    assert by_id["992654"] is True
    # Knee XR prior should not be relevant.
    assert by_id["777777"] is False


def test_root_post_alias():
    client = TestClient(app)
    r = client.post("/", json=EXAMPLE)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "classifier_available" in r.json()
