"""
Feature engineering for member-enrollment anomaly detection.

Input:  a pandas DataFrame whose columns match the raw CSV column names.
Output: a numeric feature matrix (numpy array) + ordered feature-name list.

Design decisions
----------------
* Pure function ``build_features(df)`` → keeps this module testable without
  any sklearn or model state.
* All categorical codes are ordinal-encoded with explicit mappings so that
  new/unseen categories fall back to 0 rather than raising a KeyError.
* Date arithmetic is done with integer arithmetic (YYYYMMDD → days-since-epoch)
  to avoid the overhead of a full datetime parse on 50k rows.
* TODO: confirm ACA dependent-age threshold with SMEs — ACA law allows
  coverage to age 26; internal systems may flag 18 as adult.  Set to 26 here.
* TODO: confirm the >5% AWD (Adjusted Weekly Demand) drop threshold with SMEs.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# ── age constants ──────────────────────────────────────────────────────────────
# TODO: SME verification needed — ACA law = 26; internal policy may differ.
_DEPENDENT_AGE_THRESHOLD = 26   # years; over this → should not be a dependent (MCDE 30–70)
_SENIOR_AGE_THRESHOLD    = 65   # years; used for SEN business-type cross-check

# ── ordinal maps (unknown → 0) ─────────────────────────────────────────────────
_STATUS_MAP       = {"0": 1, "1": 2, "2": 3, "4": 4}            # STS: active/inactive/suspended/terminated
_MBUTY_MAP        = {"IND": 1, "SMGRP": 2, "SEN": 3}
_TYP_MAP          = {"MED": 1, "DEN": 2, "VIS": 3, "LFE": 4, "STD": 5, "LTD": 6}
_TP1_MAP          = {"PPO": 1, "HMO": 2, "POS": 3, "EPO": 4}
_STATE_MAP        = {
    "CO": 1, "FL": 2, "GA": 3, "IL": 4, "MA": 5,
    "MI": 6, "MN": 7, "NC": 8, "OH": 9, "PA": 10, "TX": 11, "WI": 12,
}
_MCDE_PRIMARY     = {10, 20}   # member codes that indicate primary subscriber
_MCDE_DEPENDENT   = {30, 40, 50, 60, 70}


class Features(NamedTuple):
    """Container returned by build_features."""
    matrix: np.ndarray        # shape (n_rows, n_features), dtype float32
    names:  list[str]         # len == n_features


def _yyyymmdd_to_days(series: pd.Series) -> pd.Series:
    """
    Convert an integer YYYYMMDD column to approximate days-since-epoch.
    Rows where the value is 0 (meaning 'not set') map to NaN.
    """
    s = series.replace(0, np.nan)
    year  = (s // 10000).astype("float32")
    month = ((s % 10000) // 100).astype("float32")
    day   = (s % 100).astype("float32")
    # Approximate: 365.25 days/year, 30.4375 days/month
    return year * 365.25 + month * 30.4375 + day


def build_features(df: pd.DataFrame) -> Features:
    """
    Transform raw member DataFrame → numeric feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the raw CSV columns (column names as shipped in the file).
        Extra or missing columns are handled gracefully (filled with 0/NaN).

    Returns
    -------
    Features
        NamedTuple with .matrix (float32 ndarray) and .names (list[str]).
    """
    n = len(df)
    out: dict[str, np.ndarray] = {}

    def _col(name: str, default=0) -> pd.Series:
        """Return df[name] or a default series when the column is absent."""
        if name in df.columns:
            return df[name]
        return pd.Series([default] * n, index=df.index)

    # ── 1. Membership count ───────────────────────────────────────────────────
    out["mcnt"]          = _col("MCNT").fillna(1).astype("float32")
    out["mcnt_gt_10"]    = (out["mcnt"] > 10).astype("float32")   # unusual family size
    out["mcnt_gt_15"]    = (out["mcnt"] > 15).astype("float32")   # very unusual

    # ── 2. Member role (MCDE) ─────────────────────────────────────────────────
    mcde                 = _col("MCDE").fillna(10).astype(int)
    out["mcde"]          = mcde.astype("float32")
    out["is_primary"]    = mcde.isin(_MCDE_PRIMARY).astype("float32")
    out["is_dependent"]  = mcde.isin(_MCDE_DEPENDENT).astype("float32")

    # ── 3. Age ────────────────────────────────────────────────────────────────
    today_days = pd.Timestamp.today().year * 365.25
    membirdt   = _yyyymmdd_to_days(_col("MEMBIRDT", 0))
    age_years  = ((today_days - membirdt) / 365.25).clip(lower=0, upper=120)
    out["age_years"]           = age_years.fillna(-1).astype("float32")
    # Flag: over-age dependent (TODO: confirm threshold with SME — ACA = 26)
    out["overage_dependent"]   = (
        (age_years > _DEPENDENT_AGE_THRESHOLD) & mcde.isin(_MCDE_DEPENDENT)
    ).astype("float32")
    out["minor_primary"]       = (
        (age_years < 18) & mcde.isin(_MCDE_PRIMARY)
    ).astype("float32")
    out["senior_non_sen_plan"] = (
        (age_years >= _SENIOR_AGE_THRESHOLD) &
        (_col("MBUTY", "").astype(str) != "SEN")
    ).astype("float32")

    # ── 4. Dates & durations ─────────────────────────────────────────────────
    cont_eff  = _yyyymmdd_to_days(_col("CONT EFF", 0))
    mem_eff   = _yyyymmdd_to_days(_col("MEMEFFDT", 0))
    cont_can  = _yyyymmdd_to_days(_col("CONT CAN", 0))   # NaN when 0 (active)
    mem_can   = _yyyymmdd_to_days(_col("MEMCANDT", 0))

    today_f   = float(today_days)
    coverage_duration = (today_f - cont_eff).clip(lower=0)
    out["coverage_duration_days"]   = coverage_duration.fillna(0).astype("float32")
    out["mem_eff_lag_days"]         = (mem_eff - cont_eff).clip(lower=0).fillna(0).astype("float32")
    out["has_cancel_date"]          = cont_can.notna().astype("float32")
    out["immediate_cancel"]         = (
        cont_can.notna() & ((cont_can - cont_eff).abs() <= 7)
    ).astype("float32")
    out["cancel_date_mismatch"]     = (
        cont_can.notna() & mem_can.notna() & ((cont_can - mem_can).abs() > 30)
    ).astype("float32")
    out["future_effective_date"]    = (cont_eff > today_f).astype("float32")

    # ── 5. Status & cancellation reason ──────────────────────────────────────
    sts_raw          = _col("STS", 0).astype(str)
    out["sts"]       = sts_raw.map(lambda x: _STATUS_MAP.get(x, 0)).astype("float32")
    out["has_cr"]    = _col("CR", "").astype(str).ne("").astype("float32")
    out["sts_active_with_cancel_date"] = (
        (sts_raw == "0") & cont_can.notna()
    ).astype("float32")

    # ── 6. Business / plan type ───────────────────────────────────────────────
    out["mbuty"] = _col("MBUTY", "").astype(str).map(lambda x: _MBUTY_MAP.get(x, 0)).astype("float32")
    out["typ"]   = _col("TYP",   "").astype(str).map(lambda x: _TYP_MAP.get(x, 0)).astype("float32")
    out["tp1"]   = _col("TP1",   "").astype(str).map(lambda x: _TP1_MAP.get(x, 0)).astype("float32")
    out["state"] = _col("ST",    "").astype(str).map(lambda x: _STATE_MAP.get(x, 0)).astype("float32")

    # ── 7. SSN missingness (data quality) ────────────────────────────────────
    out["missing_ssn"] = (
        _col("SSN", "").astype(str).isin(["", "nan", "NaN", "None"])
        | _col("SSN", "").isnull()
    ).astype("float32")

    # ── 8. Primary/child (individual IND with MCDE suggesting child) ─────────
    ind_mask           = _col("MBUTY", "").astype(str) == "IND"
    out["ind_multi_member"] = (ind_mask & (out["mcnt"] > 1)).astype("float32")

    # ── 9. Premium tier ───────────────────────────────────────────────────────
    out["pre"] = _col("PRE", 0).fillna(0).clip(lower=0).astype("float32")

    # ── 10. Broker / provider presence ───────────────────────────────────────
    out["has_broker"]  = _col("BROKER", "").astype(str).ne("").astype("float32")
    out["has_provider"] = _col("PRVDR", "").astype(str).ne("").astype("float32")

    # ── assemble matrix ───────────────────────────────────────────────────────
    names  = list(out.keys())
    matrix = np.column_stack([out[k].values for k in names]).astype(np.float32)

    return Features(matrix=matrix, names=names)


# Exposed constant so other modules can reference the feature count.
FEATURE_NAMES: list[str] = list(build_features(
    pd.DataFrame({c: [0] for c in [
        "MCNT", "MCDE", "MEMBIRDT", "CONT EFF", "MEMEFFDT",
        "CONT CAN", "MEMCANDT", "STS", "CR", "MBUTY",
        "TYP", "TP1", "ST", "SSN", "PRE", "BROKER", "PRVDR",
    ]})
).names)
