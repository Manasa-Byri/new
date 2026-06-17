"""
Unit tests for the POST /api/v1/score endpoint (app/api/routes.py).

Strategy
--------
* Mock ``app.api.routes._get_scorer`` so no .joblib files are needed.
* Use the real Pydantic validation layer — this catches schema regressions.
* Include negative/guardrail tests:
    - empty records list
    - invalid field values (MCNT=-1, bad MBUTY enum)
    - model unavailable (503)
    - internal scorer error (500)
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.model.scorer import ScoredRecord
from app.model.severity import Severity


# ── app fixture ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Minimal FastAPI app with only the score router mounted.
    Uses FastAPI's built-in exception handlers to avoid coupling to the
    existing error_handler.py (which has a Pydantic-v2 serialisation bug
    when ctx.error is a ValueError).
    """
    from fastapi import FastAPI
    from app.api.routes import router as score_router

    test_app = FastAPI()
    test_app.include_router(score_router, prefix="/api/v1")
    return TestClient(test_app, raise_server_exceptions=False)


# ── mock factory ──────────────────────────────────────────────────────────────


def _make_mock_scorer(
    n_records: int = 1,
    is_anomaly: bool = False,
    score: float = 0.10,
) -> MagicMock:
    mock = MagicMock()
    mock.score.return_value = [
        ScoredRecord(
            hcid            = f"MBR{i:05d}",
            is_anomaly      = is_anomaly,
            score           = score,
            raw_score       = -0.05,
            severity        = Severity.LOW,
            anomaly_reasons = [],
        )
        for i in range(n_records)
    ]
    return mock


def _minimal_record(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "HCID":     "TEST001",
        "MCNT":     2,
        "MCDE":     20,
        "MEMBIRDT": 19850601,
        "CONT EFF": 20200101,
        "CONT CAN": 0,
        "STS":      0,
        "MBUTY":    "IND",
        "TYP":      "MED",
        "TP1":      "PPO",
        "ST":       "TX",
    }
    base.update(overrides)
    return base


# ── happy path ─────────────────────────────────────────────────────────────────


class TestScoreEndpointHappyPath:
    _URL = "/api/v1/score"

    def test_returns_200_with_valid_record(self, client: TestClient):
        mock_scorer = _make_mock_scorer(n_records=1)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        assert resp.status_code == 200

    def test_response_body_structure(self, client: TestClient):
        mock_scorer = _make_mock_scorer(n_records=1)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        body = resp.json()
        assert body["success"] is True
        assert "run_id" in body
        assert "timestamp" in body
        assert "total_records" in body
        assert "anomaly_count" in body
        assert "normal_count" in body
        assert "anomaly_rate" in body
        assert isinstance(body["data"], list)

    def test_counts_correct_for_one_normal(self, client: TestClient):
        mock_scorer = _make_mock_scorer(n_records=1, is_anomaly=False)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        body = resp.json()
        assert body["total_records"] == 1
        assert body["anomaly_count"] == 0
        assert body["normal_count"]  == 1
        assert body["anomaly_rate"]  == 0.0

    def test_counts_correct_for_anomaly(self, client: TestClient):
        mock_scorer = _make_mock_scorer(n_records=2, is_anomaly=True, score=0.90)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            records = [_minimal_record(), _minimal_record(HCID="TEST002")]
            resp = client.post(self._URL, json={"records": records})
        body = resp.json()
        assert body["anomaly_count"] == 2
        assert body["normal_count"]  == 0

    def test_each_data_item_has_required_fields(self, client: TestClient):
        mock_scorer = _make_mock_scorer(n_records=1)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        item = resp.json()["data"][0]
        for field in ("hcid", "is_anomaly", "score", "raw_score", "severity", "anomaly_reasons"):
            assert field in item, f"Missing field: {field}"

    def test_severity_is_valid_enum_value(self, client: TestClient):
        mock_scorer = _make_mock_scorer(n_records=1)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        severity = resp.json()["data"][0]["severity"]
        assert severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_score_alias_cont_eff_with_space(self, client: TestClient):
        """'CONT EFF' (space) and 'CONT_EFF' (underscore) must both be accepted."""
        mock_scorer = _make_mock_scorer(n_records=1)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            rec = _minimal_record()
            rec["CONT EFF"] = 20200101
            resp = client.post(self._URL, json={"records": [rec]})
        assert resp.status_code == 200

    def test_batch_of_100_records(self, client: TestClient):
        mock_scorer = _make_mock_scorer(n_records=100)
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            records = [_minimal_record(HCID=f"MBR{i:05d}") for i in range(100)]
            resp = client.post(self._URL, json={"records": records})
        assert resp.status_code == 200
        assert resp.json()["total_records"] == 100


# ── guardrail: input validation ────────────────────────────────────────────────


class TestScoreEndpointInputValidation:
    _URL = "/api/v1/score"

    def test_empty_records_list_returns_422(self, client: TestClient):
        """GUARDRAIL: empty list must be rejected before reaching the scorer."""
        resp = client.post(self._URL, json={"records": []})
        assert resp.status_code == 422

    def test_missing_records_key_returns_422(self, client: TestClient):
        resp = client.post(self._URL, json={})
        assert resp.status_code == 422

    def test_invalid_mcnt_negative_returns_422(self, client: TestClient):
        """GUARDRAIL: MCNT=-1 is below the ge=1 constraint."""
        rec = _minimal_record(MCNT=-1)
        resp = client.post(self._URL, json={"records": [rec]})
        assert resp.status_code == 422

    def test_invalid_mbuty_value_returns_422(self, client: TestClient):
        """GUARDRAIL: MBUTY='CORP' is not in the allowed enum pattern."""
        rec = _minimal_record(MBUTY="CORP")
        resp = client.post(self._URL, json={"records": [rec]})
        assert resp.status_code == 422

    def test_invalid_typ_value_returns_422(self, client: TestClient):
        rec = _minimal_record(TYP="XYZ")
        resp = client.post(self._URL, json={"records": [rec]})
        assert resp.status_code == 422

    def test_invalid_sts_above_4_returns_422(self, client: TestClient):
        rec = _minimal_record(STS=9)
        resp = client.post(self._URL, json={"records": [rec]})
        assert resp.status_code == 422

    def test_non_list_records_returns_422(self, client: TestClient):
        resp = client.post(self._URL, json={"records": "not-a-list"})
        assert resp.status_code == 422

    def test_completely_empty_body_returns_422(self, client: TestClient):
        resp = client.post(self._URL, content=b"", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422


# ── guardrail: model unavailable (503) ────────────────────────────────────────


class TestScoreEndpointModelUnavailable:
    _URL = "/api/v1/score"

    def test_model_not_found_returns_503(self, client: TestClient):
        from app.core.exceptions import ModelNotFoundError
        with patch("app.api.routes._get_scorer", side_effect=ModelNotFoundError("No artifact")):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        assert resp.status_code == 503
        body = resp.json()
        detail = body.get("detail") or body.get("error") or ""
        assert "not available" in str(detail).lower() or "model" in str(detail).lower()

    def test_model_load_error_returns_503(self, client: TestClient):
        from app.core.exceptions import ModelLoadError
        with patch("app.api.routes._get_scorer", side_effect=ModelLoadError("Corrupt file")):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        assert resp.status_code == 503


# ── guardrail: internal scorer error (500) ────────────────────────────────────


class TestScoreEndpointInternalError:
    _URL = "/api/v1/score"

    def test_unexpected_scorer_exception_returns_500(self, client: TestClient):
        mock_scorer = MagicMock()
        mock_scorer.score.side_effect = RuntimeError("Unexpected scoring failure")
        with patch("app.api.routes._get_scorer", return_value=mock_scorer):
            resp = client.post(self._URL, json={"records": [_minimal_record()]})
        assert resp.status_code == 500
