"""
Unit tests for app/model/features.py

Tests are hermetic — no filesystem, network, or database access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model.features import FEATURE_NAMES, build_features


# ── helpers ────────────────────────────────────────────────────────────────────


def _base_row(**overrides) -> dict:
    """Return a minimal valid member row, overriding specific fields."""
    today = 20260101
    row = {
        "HCID": "TEST001",
        "SSN": "123456789",
        "MCNT": 2,
        "MCDE": 20,
        "MEMBIRDT": 19850601,   # ~40 years old → adult
        "CONT EFF": 20200101,
        "MEMEFFDT": 20200101,
        "CONT CAN": 0,
        "MEMCANDT": 0,
        "STS": 0,
        "CR": "",
        "MBUTY": "IND",
        "TYP": "MED",
        "TP1": "PPO",
        "ST": "TX",
        "SSN": "123456789",
        "PRE": 5.0,
        "BROKER": "",
        "PRVDR": "001215Y",
    }
    row.update(overrides)
    return row


def _df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── shape contract ─────────────────────────────────────────────────────────────


class TestBuildFeaturesShape:
    def test_returns_named_tuple(self):
        features = build_features(_df(_base_row()))
        assert hasattr(features, "matrix")
        assert hasattr(features, "names")

    def test_matrix_dtype_float32(self):
        features = build_features(_df(_base_row()))
        assert features.matrix.dtype == np.float32

    def test_names_length_matches_matrix_columns(self):
        features = build_features(_df(_base_row()))
        assert len(features.names) == features.matrix.shape[1]

    def test_matrix_row_count_matches_input(self):
        df = _df(_base_row(), _base_row(MCNT=5), _base_row(MCNT=10))
        features = build_features(df)
        assert features.matrix.shape[0] == 3

    def test_feature_names_constant_matches_build(self):
        """FEATURE_NAMES module constant must equal the names from a live call."""
        features = build_features(_df(_base_row()))
        assert features.names == FEATURE_NAMES

    def test_empty_dataframe_returns_zero_rows(self):
        df = pd.DataFrame(columns=list(_base_row().keys()))
        features = build_features(df)
        assert features.matrix.shape[0] == 0
        assert len(features.names) > 0

    def test_missing_columns_do_not_raise(self):
        """Columns absent from the input should default gracefully."""
        df = pd.DataFrame([{"MCNT": 2}])
        features = build_features(df)
        assert features.matrix.shape == (1, len(features.names))


# ── SSN missingness flag ───────────────────────────────────────────────────────


class TestMissingSSN:
    def _get(self, df: pd.DataFrame, feature: str) -> np.ndarray:
        features = build_features(df)
        idx = features.names.index(feature)
        return features.matrix[:, idx]

    def test_null_ssn_flags_missing(self):
        row = _base_row()
        row["SSN"] = None
        arr = self._get(_df(row), "missing_ssn")
        assert arr[0] == 1.0

    def test_empty_string_ssn_flags_missing(self):
        arr = self._get(_df(_base_row(SSN="")), "missing_ssn")
        assert arr[0] == 1.0

    def test_valid_ssn_does_not_flag(self):
        arr = self._get(_df(_base_row(SSN="123456789")), "missing_ssn")
        assert arr[0] == 0.0


# ── overage dependent ──────────────────────────────────────────────────────────


class TestOverageDependent:
    def _get(self, df: pd.DataFrame) -> np.ndarray:
        features = build_features(df)
        idx = features.names.index("overage_dependent")
        return features.matrix[:, idx]

    def test_adult_dependent_flagged(self):
        """MCDE=30 (dependent) + birth 1990 → over-26 dependent."""
        row = _base_row(MCDE=30, MEMBIRDT=19900101)
        arr = self._get(_df(row))
        assert arr[0] == 1.0

    def test_young_dependent_not_flagged(self):
        """MCDE=30 + birth 2005 → under-26, should not flag."""
        row = _base_row(MCDE=30, MEMBIRDT=20051001)
        arr = self._get(_df(row))
        assert arr[0] == 0.0

    def test_adult_primary_not_flagged(self):
        """MCDE=20 (primary) at any age → not overage_dependent."""
        row = _base_row(MCDE=20, MEMBIRDT=19800101)
        arr = self._get(_df(row))
        assert arr[0] == 0.0


# ── IND multi-member ────────────────────────────────────────────────────────────


class TestIndMultiMember:
    def _get(self, df: pd.DataFrame) -> np.ndarray:
        features = build_features(df)
        idx = features.names.index("ind_multi_member")
        return features.matrix[:, idx]

    def test_ind_with_mcnt_2_flagged(self):
        arr = self._get(_df(_base_row(MBUTY="IND", MCNT=2)))
        assert arr[0] == 1.0

    def test_ind_solo_not_flagged(self):
        arr = self._get(_df(_base_row(MBUTY="IND", MCNT=1)))
        assert arr[0] == 0.0

    def test_smgrp_with_mcnt_2_not_flagged(self):
        arr = self._get(_df(_base_row(MBUTY="SMGRP", MCNT=2)))
        assert arr[0] == 0.0


# ── immediate cancellation ─────────────────────────────────────────────────────


class TestImmediateCancel:
    def _get(self, df: pd.DataFrame) -> np.ndarray:
        features = build_features(df)
        idx = features.names.index("immediate_cancel")
        return features.matrix[:, idx]

    def test_cancel_same_day_flagged(self):
        row = _base_row(**{"CONT EFF": 20250101, "CONT CAN": 20250101})
        arr = self._get(_df(row))
        assert arr[0] == 1.0

    def test_cancel_3_days_later_flagged(self):
        row = _base_row(**{"CONT EFF": 20250101, "CONT CAN": 20250104})
        arr = self._get(_df(row))
        assert arr[0] == 1.0

    def test_cancel_30_days_later_not_flagged(self):
        row = _base_row(**{"CONT EFF": 20250101, "CONT CAN": 20250201})
        arr = self._get(_df(row))
        assert arr[0] == 0.0

    def test_active_member_not_flagged(self):
        row = _base_row(**{"CONT EFF": 20250101, "CONT CAN": 0})
        arr = self._get(_df(row))
        assert arr[0] == 0.0


# ── active with cancel date ────────────────────────────────────────────────────


class TestActiveWithCancelDate:
    def _get(self, df: pd.DataFrame) -> np.ndarray:
        features = build_features(df)
        idx = features.names.index("sts_active_with_cancel_date")
        return features.matrix[:, idx]

    def test_active_status_with_cancel_flagged(self):
        row = _base_row(STS=0, **{"CONT CAN": 20251231})
        arr = self._get(_df(row))
        assert arr[0] == 1.0

    def test_active_status_no_cancel_not_flagged(self):
        row = _base_row(STS=0, **{"CONT CAN": 0})
        arr = self._get(_df(row))
        assert arr[0] == 0.0

    def test_terminated_with_cancel_not_flagged(self):
        row = _base_row(STS=4, **{"CONT CAN": 20251231})
        arr = self._get(_df(row))
        assert arr[0] == 0.0


# ── large family ───────────────────────────────────────────────────────────────


class TestLargeFamily:
    def test_mcnt_gt_15(self):
        features = build_features(_df(_base_row(MCNT=16)))
        idx = features.names.index("mcnt_gt_15")
        assert features.matrix[0, idx] == 1.0

    def test_mcnt_10_not_gt_15(self):
        features = build_features(_df(_base_row(MCNT=10)))
        idx = features.names.index("mcnt_gt_15")
        assert features.matrix[0, idx] == 0.0


# ── numerical ranges — no NaN / Inf in output ─────────────────────────────────


class TestNumericalSanity:
    def test_no_inf_in_matrix(self):
        rows = [_base_row(MCNT=i % 20 + 1) for i in range(50)]
        features = build_features(_df(*rows))
        assert not np.any(np.isinf(features.matrix))

    def test_coverage_duration_non_negative(self):
        row = _base_row(**{"CONT EFF": 20200101})
        features = build_features(_df(row))
        idx = features.names.index("coverage_duration_days")
        assert features.matrix[0, idx] >= 0.0
