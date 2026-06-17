"""
Unit tests for app/model/severity.py

Tests are hermetic — no IO, no model loading.
"""
from __future__ import annotations

import pytest

from app.model.severity import Severity, classify_severity


class TestClassifySeverity:
    # ── happy path ─────────────────────────────────────────────────────────
    def test_score_1_is_critical(self):
        assert classify_severity(1.0) == Severity.CRITICAL

    def test_score_at_critical_threshold_is_critical(self):
        assert classify_severity(0.80) == Severity.CRITICAL

    def test_score_just_below_critical_is_high(self):
        assert classify_severity(0.799) == Severity.HIGH

    def test_score_at_high_threshold_is_high(self):
        assert classify_severity(0.60) == Severity.HIGH

    def test_score_just_below_high_is_medium(self):
        assert classify_severity(0.599) == Severity.MEDIUM

    def test_score_at_medium_threshold_is_medium(self):
        assert classify_severity(0.40) == Severity.MEDIUM

    def test_score_just_below_medium_is_low(self):
        assert classify_severity(0.399) == Severity.LOW

    def test_score_0_is_low(self):
        assert classify_severity(0.0) == Severity.LOW

    # ── custom thresholds ─────────────────────────────────────────────────
    def test_custom_thresholds_respected(self):
        assert classify_severity(0.90, critical=0.95, high=0.70, medium=0.50) == Severity.HIGH
        assert classify_severity(0.96, critical=0.95, high=0.70, medium=0.50) == Severity.CRITICAL

    # ── guardrail: invalid input ──────────────────────────────────────────
    def test_score_above_1_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            classify_severity(1.001)

    def test_score_below_0_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            classify_severity(-0.001)

    # ── enum string value ─────────────────────────────────────────────────
    def test_severity_values_are_strings(self):
        """Severity must serialise to string for JSON responses."""
        assert Severity.CRITICAL.value == "CRITICAL"
        assert Severity.HIGH.value == "HIGH"
        assert Severity.MEDIUM.value == "MEDIUM"
        assert Severity.LOW.value == "LOW"
