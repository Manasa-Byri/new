"""
Unit tests for app/model/scorer.py — Scorer class.

All tests are hermetic: the model is fit() on tiny synthetic DataFrames.
No disk access, no network, no database.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import ModelNotFoundError, ModelNotTrainedError
from app.model.scorer import Scorer, ScoredRecord, _explain_row
from app.model.severity import Severity


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_df(n: int = 50, *, seed: int = 42) -> pd.DataFrame:
    """
    Produce a synthetic member DataFrame large enough to fit IsolationForest.
    All rows represent plausible normal members; the last row is seeded with
    obvious anomaly signals (immediate cancel, large family, missing SSN).
    """
    rng = np.random.default_rng(seed)

    # fmt: off
    rows = []
    for i in range(n - 1):
        rows.append({
            "HCID":     f"MBR{i:05d}",
            "SSN":      f"{rng.integers(100000000, 999999999)}",
            "MCNT":     int(rng.integers(1, 5)),
            "MCDE":     int(rng.choice([20, 30])),
            "MEMBIRDT": int(rng.integers(19600101, 20000101)),
            "CONT EFF": 20200101,
            "MEMEFFDT": 20200101,
            "CONT CAN": 0,
            "MEMCANDT": 0,
            "STS":      0,
            "CR":       "",
            "MBUTY":    "SMGRP",
            "TYP":      "MED",
            "TP1":      "PPO",
            "ST":       "TX",
            "PRE":      5.0,
            "BROKER":   "",
            "PRVDR":    "001215Y",
        })
    # Last row: obvious anomaly signals
    rows.append({
        "HCID":     "ANOMALY01",
        "SSN":      None,            # MISSING_SSN
        "MCNT":     20,              # UNUSUALLY_LARGE_FAMILY
        "MCDE":     30,
        "MEMBIRDT": 19500101,        # OVERAGE_DEPENDENT
        "CONT EFF": 20250101,
        "MEMEFFDT": 20250101,
        "CONT CAN": 20250102,        # IMMEDIATE_CANCEL
        "MEMCANDT": 0,
        "STS":      0,               # ACTIVE_WITH_CANCEL_DATE
        "CR":       "01",
        "MBUTY":    "IND",
        "TYP":      "MED",
        "TP1":      "PPO",
        "ST":       "TX",
        "PRE":      0.0,
        "BROKER":   "",
        "PRVDR":    "",
    })
    # fmt: on
    return pd.DataFrame(rows)


@pytest.fixture()
def fitted_scorer(tmp_path: Path) -> Scorer:
    """Return a Scorer that has been fit() on synthetic data."""
    scorer = Scorer()
    scorer.fit(_make_df(60))
    return scorer


# ── fit / score contract ───────────────────────────────────────────────────────


class TestScorerFit:
    def test_fit_returns_self(self):
        scorer = Scorer()
        result = scorer.fit(_make_df(60))
        assert result is scorer

    def test_fit_sets_model_and_scaler(self):
        scorer = Scorer()
        scorer.fit(_make_df(60))
        assert scorer._model is not None
        assert scorer._scaler is not None


class TestScorerScore:
    def test_score_returns_list_of_scored_records(self, fitted_scorer: Scorer):
        df      = _make_df(5)
        results = fitted_scorer.score(df)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, ScoredRecord) for r in results)

    def test_score_outputs_valid_score_range(self, fitted_scorer: Scorer):
        for r in fitted_scorer.score(_make_df(20)):
            assert 0.0 <= r.score <= 1.0, f"Out-of-range score: {r.score}"

    def test_score_severity_matches_score(self, fitted_scorer: Scorer):
        from app.model.severity import classify_severity
        from app.config.settings import get_settings
        s = get_settings()
        for r in fitted_scorer.score(_make_df(20)):
            expected = classify_severity(
                r.score,
                critical=s.SEVERITY_CRITICAL_THRESHOLD,
                high=s.SEVERITY_HIGH_THRESHOLD,
                medium=s.SEVERITY_MEDIUM_THRESHOLD,
            )
            assert r.severity == expected

    def test_score_is_anomaly_is_bool(self, fitted_scorer: Scorer):
        for r in fitted_scorer.score(_make_df(10)):
            assert isinstance(r.is_anomaly, bool)

    def test_score_preserves_row_order(self, fitted_scorer: Scorer):
        df = _make_df(10)
        results = fitted_scorer.score(df)
        for i, r in enumerate(results):
            if r.hcid:
                assert r.hcid == df.iloc[i]["HCID"]

    def test_score_single_record(self, fitted_scorer: Scorer):
        df      = pd.DataFrame([_make_df(1).iloc[0].to_dict()])
        results = fitted_scorer.score(df)
        assert len(results) == 1


# ── guardrail: score before fit ────────────────────────────────────────────────


class TestScorerGuardrails:
    def test_score_before_fit_raises(self):
        scorer = Scorer()
        with pytest.raises(ModelNotTrainedError):
            scorer.score(_make_df(5))

    def test_save_before_fit_raises(self, tmp_path: Path):
        scorer = Scorer()
        with pytest.raises(ModelNotTrainedError):
            scorer.save(tmp_path)

    def test_load_missing_files_raises(self, tmp_path: Path):
        scorer = Scorer()
        with pytest.raises(ModelNotFoundError):
            scorer.load(tmp_path)


# ── save / load roundtrip ──────────────────────────────────────────────────────


class TestScorerPersistence:
    def test_save_creates_artifacts(self, fitted_scorer: Scorer, tmp_path: Path):
        fitted_scorer.save(tmp_path)
        assert (tmp_path / "v1_isolation_forest.joblib").exists()
        assert (tmp_path / "v1_preprocessor.joblib").exists()

    def test_load_restores_scoring_capability(self, fitted_scorer: Scorer, tmp_path: Path):
        fitted_scorer.save(tmp_path)
        loaded = Scorer()
        loaded.load(tmp_path)
        results = loaded.score(_make_df(5))
        assert len(results) == 5

    def test_save_load_produces_identical_scores(self, fitted_scorer: Scorer, tmp_path: Path):
        df            = _make_df(10)
        scores_before = [r.score for r in fitted_scorer.score(df)]
        fitted_scorer.save(tmp_path)
        loaded        = Scorer()
        loaded.load(tmp_path)
        scores_after  = [r.score for r in loaded.score(df)]
        assert scores_before == scores_after, "Scores differ after save/load roundtrip"


# ── rule-based explainer ───────────────────────────────────────────────────────


class TestExplainRow:
    def _row(self, **kwargs) -> pd.Series:
        base = {
            "HCID": "X", "SSN": "123456789", "MCNT": 2, "MCDE": 20,
            "MEMBIRDT": 19850601, "CONT EFF": 20200101, "CONT CAN": 0,
            "MEMCANDT": 0, "MEMEFFDT": 20200101, "STS": 0, "MBUTY": "SMGRP",
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_missing_ssn_detected(self):
        reasons = _explain_row(self._row(SSN=None))
        assert "MISSING_SSN" in reasons

    def test_overage_dependent_detected(self):
        reasons = _explain_row(self._row(MCDE=30, MEMBIRDT=19900101))
        assert "OVERAGE_DEPENDENT" in reasons

    def test_ind_multi_member_detected(self):
        reasons = _explain_row(self._row(MBUTY="IND", MCNT=3))
        assert "IND_MULTI_MEMBER" in reasons

    def test_immediate_cancel_detected(self):
        reasons = _explain_row(self._row(**{"CONT EFF": 20250101, "CONT CAN": 20250102}))
        assert "IMMEDIATE_CANCEL" in reasons

    def test_active_with_cancel_detected(self):
        reasons = _explain_row(self._row(STS=0, **{"CONT CAN": 20251231}))
        assert "ACTIVE_WITH_CANCEL_DATE" in reasons

    def test_large_family_detected(self):
        reasons = _explain_row(self._row(MCNT=20))
        assert "UNUSUALLY_LARGE_FAMILY" in reasons

    def test_normal_row_has_no_reasons(self):
        reasons = _explain_row(self._row(SSN="123456789", MCNT=2, MBUTY="SMGRP"))
        assert reasons == []
