"""
Scorer — the core ML class for v1 of the AI Insight Agent.

Public API
----------
    scorer = Scorer()
    scorer.fit(df)               # train IsolationForest on a DataFrame
    scorer.save(model_dir)       # persist to two .joblib files
    scorer.load(model_dir)       # restore from disk
    results = scorer.score(df)   # returns list[ScoredRecord]

Design decisions
----------------
* IsolationForest is chosen over deep-learning approaches because:
  - Fully unsupervised: no labelled fraud/anomaly examples exist.
  - Handles mixed tabular data (numeric + encoded categoricals) natively.
  - Trains in seconds; scales to millions of rows; no GPU required.
  - Reproducible: fixed random_state=42.
  - Decision function maps directly to interpretable anomaly scores.
  
  Rejected: Autoencoder (over-engineering for v1 tabular data; needs GPU,
  long training, opaque reconstruction loss), One-Class SVM (quadratic
  complexity O(n²) — too slow at 50k+ rows).

* StandardScaler is applied inside fit() / transform() here, NOT inside
  features.py, so the fitted scaler is saved alongside the model and reused
  identically at inference time — prevents train/serve skew.

* save() / load() are symmetric operations using two separate .joblib files
  (model + preprocessor) so either can be retrained independently.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.config.settings import get_settings
from app.core.exceptions import ModelLoadError, ModelNotFoundError, ModelNotTrainedError
from app.core.logging import get_logger
from app.model.features import FEATURE_NAMES, build_features
from app.model.severity import Severity, classify_severity

logger = get_logger(__name__)


# ── output dataclass ──────────────────────────────────────────────────────────


@dataclass
class ScoredRecord:
    """
    Result for a single member record returned by Scorer.score().

    Attributes
    ----------
    hcid            : member identifier (may be None if missing in input).
    is_anomaly      : True when IsolationForest labels this record as an outlier.
    score           : normalised anomaly score in [0, 1]; higher = more anomalous.
    raw_score       : raw IsolationForest decision_function value (negative = anomaly).
    severity        : CRITICAL | HIGH | MEDIUM | LOW (based on score thresholds).
    anomaly_reasons : human-readable list of triggered rule-based flags.
    """
    hcid:            Optional[str]
    is_anomaly:      bool
    score:           float          # [0, 1]
    raw_score:       float
    severity:        Severity
    anomaly_reasons: list[str] = field(default_factory=list)


# ── scorer ────────────────────────────────────────────────────────────────────


class Scorer:
    """
    Isolation-Forest anomaly scorer for health-insurance enrollment records.

    Lifecycle
    ---------
    Training time:  fit(df) → save(model_dir)
    Serving time:   load(model_dir) → score(df)
    """

    _MODEL_FILENAME:  str = "v1_isolation_forest.joblib"
    _SCALER_FILENAME: str = "v1_preprocessor.joblib"

    def __init__(self) -> None:
        self._settings  = get_settings()
        self._model:    Optional[IsolationForest] = None
        self._scaler:   Optional[StandardScaler]  = None
        self._feature_names: list[str]            = FEATURE_NAMES

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "Scorer":
        """
        Train the Isolation Forest on a member DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw member data.  Must contain at minimum the columns listed in
            app.model.features.FEATURE_NAMES.

        Returns
        -------
        self
        """
        run_id = str(uuid.uuid4())[:8]
        logger.info("Scorer.fit started", extra={"run_id": run_id, "rows": len(df)})
        t0 = time.perf_counter()

        s = self._settings
        features = build_features(df)
        X = features.matrix  # float32, shape (n, n_features)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            n_estimators=s.N_ESTIMATORS,
            contamination=s.CONTAMINATION,
            max_samples=s.MAX_SAMPLES,
            random_state=s.RANDOM_STATE,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        self._feature_names = features.names

        elapsed = time.perf_counter() - t0
        logger.info(
            "Scorer.fit completed",
            extra={"run_id": run_id, "elapsed_s": round(elapsed, 2), "features": len(features.names)},
        )
        return self

    def save(self, model_dir: Optional[Path] = None) -> Path:
        """
        Persist the fitted model and scaler to ``model_dir``.

        Returns
        -------
        Path
            The directory where artifacts were saved.
        """
        self._require_fitted()
        d = Path(model_dir or self._settings.MODEL_DIR)
        d.mkdir(parents=True, exist_ok=True)

        model_path  = d / self._MODEL_FILENAME
        scaler_path = d / self._SCALER_FILENAME

        joblib.dump(self._model,  model_path)
        joblib.dump(self._scaler, scaler_path)

        logger.info("Model artifacts saved", extra={"dir": str(d)})
        return d

    def load(self, model_dir: Optional[Path] = None) -> "Scorer":
        """
        Load a previously fitted model and scaler from ``model_dir``.

        Returns
        -------
        self
        """
        d = Path(model_dir or self._settings.MODEL_DIR)
        model_path  = d / self._MODEL_FILENAME
        scaler_path = d / self._SCALER_FILENAME

        for p in (model_path, scaler_path):
            if not p.exists():
                raise ModelNotFoundError(f"Model artifact not found: {p}")

        try:
            self._model  = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to deserialise model from {d}: {exc}") from exc

        logger.info("Model artifacts loaded", extra={"dir": str(d)})
        return self

    def score(self, df: pd.DataFrame) -> list[ScoredRecord]:
        """
        Score a batch of member records.

        Parameters
        ----------
        df : pd.DataFrame
            Raw member data.  Columns not present in the feature set are
            gracefully ignored.

        Returns
        -------
        list[ScoredRecord]
            One entry per row in df, in the same order.
        """
        self._require_fitted()
        run_id = str(uuid.uuid4())[:8]
        logger.info("Scorer.score started", extra={"run_id": run_id, "rows": len(df)})
        t0 = time.perf_counter()

        s        = self._settings
        features = build_features(df)
        X        = features.matrix
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]

        labels     = self._model.predict(X_scaled)           # type: ignore[union-attr]
        raw_scores = self._model.decision_function(X_scaled) # type: ignore[union-attr]

        offset = float(self._model.offset_)  # type: ignore[union-attr]
        scale  = max(float(np.abs(raw_scores).max()), abs(offset), 0.5)
        norm_scores = np.clip(0.5 - raw_scores / (2.0 * scale), 0.0, 1.0)

        hcids = df.get("HCID", pd.Series([None] * len(df))).tolist()

        # ── vectorised business-rule flags ────────────────────────────────────
        rule_flags = _explain_batch(df, labels)

        results: list[ScoredRecord] = []
        for i in range(len(df)):
            is_anomaly = bool(labels[i] == -1)
            score      = float(round(float(norm_scores[i]), 6))
            raw_score  = float(round(float(raw_scores[i]), 6))
            severity   = classify_severity(
                score,
                critical=s.SEVERITY_CRITICAL_THRESHOLD,
                high=s.SEVERITY_HIGH_THRESHOLD,
                medium=s.SEVERITY_MEDIUM_THRESHOLD,
            )
            results.append(ScoredRecord(
                hcid            = str(hcids[i]) if hcids[i] and str(hcids[i]) not in ("nan", "None") else None,
                is_anomaly      = is_anomaly,
                score           = score,
                raw_score       = raw_score,
                severity        = severity,
                anomaly_reasons = rule_flags[i] if is_anomaly else [],
            ))

        elapsed = time.perf_counter() - t0
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "Scorer.score completed",
            extra={
                "run_id":        run_id,
                "elapsed_s":     round(elapsed, 2),
                "total":         len(results),
                "anomalies":     anomaly_count,
                "anomaly_rate":  round(anomaly_count / max(len(results), 1), 4),
            },
        )
        return results

    # ── private helpers ───────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if self._model is None or self._scaler is None:
            raise ModelNotTrainedError(
                "Scorer has not been fitted.  Call fit() or load() first."
            )


# ── rule-based explainer (vectorised batch) ──────────────────────────────────


def _explain_batch(df: pd.DataFrame, labels: np.ndarray) -> list[list[str]]:
    """
    Compute rule flags for every row in one vectorised pass.
    Much faster than calling _explain_row(df.iloc[i]) in a Python loop.
    Returns a list of reason lists — one per row.
    """
    import datetime
    n = len(df)
    today_year = datetime.date.today().year
    today_int  = int(datetime.date.today().strftime("%Y%m%d"))

    def _col(name: str, default=None) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([default] * n, index=df.index)

    def _int_col(name: str, default: int = 0) -> pd.Series:
        s = _col(name, default)
        return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)

    def _str_col(name: str, default: str = "") -> pd.Series:
        return _col(name, default).fillna(default).astype(str).str.strip()

    # ── pre-compute columns once ──────────────────────────────────────────────
    ssn      = _str_col("SSN")
    mcde     = _int_col("MCDE", 10)
    membirdt = _int_col("MEMBIRDT", 0)
    mcnt     = _int_col("MCNT", 1)
    mbuty    = _str_col("MBUTY")
    cont_eff = _int_col("CONT EFF", 0)
    cont_can = _int_col("CONT CAN", 0)
    memeffdt = _int_col("MEMEFFDT", 0)
    memcandt = _int_col("MEMCANDT", 0)
    sts      = _int_col("STS", 0)
    pre      = pd.to_numeric(_col("PRE", -1), errors="coerce").fillna(-1)

    birth_year = (membirdt // 10000).where(membirdt > 0, other=0)
    age        = (today_year - birth_year).where(birth_year > 0, other=0)

    is_dependent = mcde.isin([30, 40, 50, 60, 70])
    is_primary   = mcde.isin([10, 20])

    def _approx_days(s: pd.Series) -> pd.Series:
        y = (s // 10000).astype(float)
        m = ((s % 10000) // 100).astype(float)
        d = (s % 100).astype(float)
        return y * 365.25 + m * 30.4375 + d

    eff_days = _approx_days(cont_eff).where(cont_eff > 0, other=np.nan)
    can_days = _approx_days(cont_can).where(cont_can > 0, other=np.nan)
    mem_days = _approx_days(memeffdt).where(memeffdt > 0, other=np.nan)
    mcan_days = _approx_days(memcandt).where(memcandt > 0, other=np.nan)

    cancel_diff = (can_days - eff_days).fillna(999)
    mcan_diff   = (can_days - mcan_days).abs().fillna(0)

    # ── boolean flag arrays ───────────────────────────────────────────────────
    FLAG_MISSING_SSN              = ssn.isin(["", "nan", "none", "NaN", "None"])
    FLAG_OVERAGE_DEPENDENT        = (age > 26) & is_dependent & (membirdt > 0)
    FLAG_MINOR_PRIMARY            = (age < 18) & (age > 0) & is_primary
    FLAG_IND_MULTI_MEMBER         = (mbuty == "IND") & (mcnt > 1)
    FLAG_IMMEDIATE_CANCEL         = (cont_can > 0) & (cont_eff > 0) & (cancel_diff >= 0) & (cancel_diff <= 7)
    FLAG_CANCEL_MISMATCH          = (cont_can > 0) & (memcandt > 0) & (mcan_diff > 30)
    FLAG_ACTIVE_WITH_CANCEL       = (sts == 0) & (cont_can > 0)
    FLAG_LARGE_FAMILY             = mcnt > 15
    FLAG_SENIOR_NON_SEN           = (age >= 65) & (age > 0) & (mbuty != "SEN") & (membirdt > 0)
    FLAG_FUTURE_EFF               = (cont_eff > 0) & (cont_eff > today_int)
    FLAG_RETRO_CANCEL             = (cont_eff > 0) & (cont_can > 0) & (cont_can < cont_eff)
    FLAG_ZERO_PREMIUM             = (sts == 0) & (pre == 0.0)
    FLAG_MISSING_BIRTH            = (membirdt == 0)
    FLAG_MEMBER_CANCEL_NO_CONTRACT= (memcandt > 0) & (cont_can == 0)
    FLAG_MEMBER_EFF_BEFORE        = (cont_eff > 0) & (memeffdt > 0) & (memeffdt < cont_eff)

    # ── build per-row reason lists only for anomalies ────────────────────────
    all_flags = [
        ("MISSING_SSN",                           FLAG_MISSING_SSN),
        ("OVERAGE_DEPENDENT",                     FLAG_OVERAGE_DEPENDENT),
        ("MINOR_PRIMARY_SUBSCRIBER",              FLAG_MINOR_PRIMARY),
        ("IND_MULTI_MEMBER",                      FLAG_IND_MULTI_MEMBER),
        ("IMMEDIATE_CANCEL",                      FLAG_IMMEDIATE_CANCEL),
        ("CANCEL_DATE_MISMATCH",                  FLAG_CANCEL_MISMATCH),
        ("ACTIVE_WITH_CANCEL_DATE",               FLAG_ACTIVE_WITH_CANCEL),
        ("UNUSUALLY_LARGE_FAMILY",                FLAG_LARGE_FAMILY),
        ("SENIOR_NON_SENIOR_PLAN",                FLAG_SENIOR_NON_SEN),
        ("FUTURE_EFFECTIVE_DATE",                 FLAG_FUTURE_EFF),
        ("RETROACTIVE_CANCEL",                    FLAG_RETRO_CANCEL),
        ("ZERO_PREMIUM_ACTIVE",                   FLAG_ZERO_PREMIUM),
        ("MISSING_BIRTH_DATE",                    FLAG_MISSING_BIRTH),
        ("MEMBER_CANCEL_WITHOUT_CONTRACT_CANCEL", FLAG_MEMBER_CANCEL_NO_CONTRACT),
        ("MEMBER_EFFECTIVE_BEFORE_CONTRACT",      FLAG_MEMBER_EFF_BEFORE),
    ]

    # Convert each boolean Series to a numpy array for fast indexing
    flag_arrays = [(name, arr.to_numpy(dtype=bool)) for name, arr in all_flags]

    result: list[list[str]] = [[] for _ in range(n)]
    anomaly_idx = np.where(labels == -1)[0]
    for i in anomaly_idx:
        result[i] = [name for name, arr in flag_arrays if arr[i]]
    return result


# ── row-level explainer (kept for single-record use) ─────────────────────────


def _explain_row(row: pd.Series) -> list[str]:
    """
    Return a list of human-readable reasons why this record looks anomalous.
    Rules are deterministic and complement the ML score with explainability.
    """
    reasons: list[str] = []

    # Missing SSN (data quality)
    ssn = str(row.get("SSN", "")).strip()
    if not ssn or ssn.lower() in ("nan", "none", ""):
        reasons.append("MISSING_SSN")

    # Over-age dependent / minor primary (TODO: confirm threshold with SME — ACA = 26)
    try:
        import datetime
        mcde     = int(float(row.get("MCDE", 10)))
        membirdt = int(float(row.get("MEMBIRDT", 0)))
        if membirdt > 0:
            birth_year = membirdt // 10000
            age = datetime.date.today().year - birth_year
            if age > 26 and mcde in {30, 40, 50, 60, 70}:
                reasons.append("OVERAGE_DEPENDENT")
            if age < 18 and mcde in {10, 20}:   # primary subscriber under 18
                reasons.append("MINOR_PRIMARY_SUBSCRIBER")
    except (ValueError, TypeError):
        pass

    # IND plan with multiple members
    try:
        mbuty = str(row.get("MBUTY", "")).strip()
        mcnt  = int(row.get("MCNT", 1))
        if mbuty == "IND" and mcnt > 1:
            reasons.append("IND_MULTI_MEMBER")
    except (ValueError, TypeError):
        pass

    # Immediate cancellation (contract cancelled within 7 days of start)
    try:
        cont_eff = int(row.get("CONT EFF", 0))
        cont_can = int(row.get("CONT CAN", 0))
        if cont_can > 0 and cont_eff > 0:
            eff_days = _yyyymmdd_to_approx_days(cont_eff)
            can_days = _yyyymmdd_to_approx_days(cont_can)
            if 0 <= can_days - eff_days <= 7:
                reasons.append("IMMEDIATE_CANCEL")
    except (ValueError, TypeError):
        pass

    # Cancel date mismatch between contract and member
    try:
        cont_can = int(row.get("CONT CAN", 0))
        mem_can  = int(row.get("MEMCANDT", 0))
        if cont_can > 0 and mem_can > 0:
            diff = abs(_yyyymmdd_to_approx_days(cont_can) - _yyyymmdd_to_approx_days(mem_can))
            if diff > 30:
                reasons.append("CANCEL_DATE_MISMATCH")
    except (ValueError, TypeError):
        pass

    # Active status but has a cancel date
    try:
        sts      = int(float(str(row.get("STS", 0)).strip() or 0))
        cont_can_raw = row.get("CONT CAN", 0)
        cont_can = int(float(str(cont_can_raw).strip() or 0)) if cont_can_raw not in (None, "", "nan", "None") else 0
        if sts == 0 and cont_can > 0:
            reasons.append("ACTIVE_WITH_CANCEL_DATE")
    except (ValueError, TypeError):
        pass

    # Very large family
    try:
        mcnt = int(row.get("MCNT", 1))
        if mcnt > 15:
            reasons.append("UNUSUALLY_LARGE_FAMILY")
    except (ValueError, TypeError):
        pass

    # Senior on non-senior plan (age ≥ 65 but MBUTY is not SEN)
    try:
        import datetime
        mbuty    = str(row.get("MBUTY", "")).strip()
        membirdt = int(float(row.get("MEMBIRDT", 0)))
        if membirdt > 0:
            birth_year = membirdt // 10000
            age = datetime.date.today().year - birth_year
            if age >= 65 and mbuty != "SEN":
                reasons.append("SENIOR_NON_SENIOR_PLAN")
    except (ValueError, TypeError):
        pass

    # Future effective date (contract has not started yet)
    try:
        import datetime
        cont_eff = int(row.get("CONT EFF", 0))
        if cont_eff > 0:
            today_int = int(datetime.date.today().strftime("%Y%m%d"))
            if cont_eff > today_int:
                reasons.append("FUTURE_EFFECTIVE_DATE")
    except (ValueError, TypeError):
        pass

    # Retroactive cancel (cancel date is before the effective date)
    try:
        cont_eff = int(row.get("CONT EFF", 0))
        cont_can = int(row.get("CONT CAN", 0))
        if cont_eff > 0 and cont_can > 0 and cont_can < cont_eff:
            reasons.append("RETROACTIVE_CANCEL")
    except (ValueError, TypeError):
        pass

    # Zero premium on an active record (possible data error or fraud)
    try:
        sts = int(float(str(row.get("STS", 0)).strip() or 0))
        pre = float(row.get("PRE", -1) or -1)
        if sts == 0 and pre == 0.0:
            reasons.append("ZERO_PREMIUM_ACTIVE")
    except (ValueError, TypeError):
        pass

    # Missing birth date (cannot verify age-based eligibility)
    try:
        membirdt = int(float(row.get("MEMBIRDT", 0)))
        if membirdt == 0:
            reasons.append("MISSING_BIRTH_DATE")
    except (ValueError, TypeError):
        reasons.append("MISSING_BIRTH_DATE")

    # Member cancel date set but contract cancel date is absent
    try:
        cont_can = int(float(str(row.get("CONT CAN", 0) or 0)))
        memcandt = int(float(str(row.get("MEMCANDT", 0) or 0)))
        if memcandt > 0 and cont_can == 0:
            reasons.append("MEMBER_CANCEL_WITHOUT_CONTRACT_CANCEL")
    except (ValueError, TypeError):
        pass

    # Member effective date is before the contract effective date
    try:
        cont_eff = int(row.get("CONT EFF", 0))
        memeffdt = int(row.get("MEMEFFDT", 0))
        if cont_eff > 0 and memeffdt > 0 and memeffdt < cont_eff:
            reasons.append("MEMBER_EFFECTIVE_BEFORE_CONTRACT")
    except (ValueError, TypeError):
        pass

    return reasons


def _yyyymmdd_to_approx_days(yyyymmdd: int) -> float:
    """Convert YYYYMMDD integer to approximate days (for delta calculations)."""
    y = yyyymmdd // 10000
    m = (yyyymmdd % 10000) // 100
    d = yyyymmdd % 100
    return y * 365.25 + m * 30.4375 + d
