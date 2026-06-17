"""
Severity classification for anomaly scores.

Thresholds are loaded from Settings so they can be tuned without code changes.
Default values (calibrated against the 49,999-row training set):

    CRITICAL  score ≥ 0.80  — strong outlier; automated alert recommended
    HIGH      score ≥ 0.60  — likely anomaly; requires analyst review
    MEDIUM    score ≥ 0.40  — borderline; queue for periodic batch review
    LOW       score <  0.40 — within expected range; informational only

Note: ``is_anomaly`` is determined by IsolationForest's contamination parameter
(default 5 %) and does NOT depend on these thresholds.  A record can have
is_anomaly=True with severity=MEDIUM (weak anomaly), or is_anomaly=False with
severity=LOW (normal).  Severity is always reported for full observability.
"""
from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


def classify_severity(
    score: float,
    *,
    critical: float = 0.80,
    high:     float = 0.60,
    medium:   float = 0.40,
) -> Severity:
    """
    Map a normalised anomaly score [0, 1] to a Severity level.

    Parameters
    ----------
    score    : normalised anomaly score in [0, 1]; higher = more anomalous.
    critical : threshold above which severity is CRITICAL.
    high     : threshold above which severity is HIGH (and below critical).
    medium   : threshold above which severity is MEDIUM (and below high).
    """
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"score must be in [0, 1]; got {score}")
    if score >= critical:
        return Severity.CRITICAL
    if score >= high:
        return Severity.HIGH
    if score >= medium:
        return Severity.MEDIUM
    return Severity.LOW
