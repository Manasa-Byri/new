"""
Insurance Member Anomaly Detection — Preprocessing + Model Pipeline
Supports: train, save, load, predict (single record or batch)
"""
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH       = MODELS_DIR / "insurance_anomaly_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "insurance_anomaly_preprocessor.joblib"
MODELS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CONTAMINATION = 0.05   # expected ~5 % anomalies
RANDOM_STATE  = 42

# Categorical columns to label-encode
CAT_COLS = ["ST", "TYP", "TP1", "MBUTY", "CR_CAT", "ETHNI"]

# Final numeric feature set fed to the model
FEATURE_COLS = [
    "AGE", "MCNT", "MCDE", "FAMILY_SIZE",
    "CONTRACT_DURATION", "DAYS_TO_CANCEL",
    "IS_CANCELLED", "IS_MEDICAL", "IS_DENTAL", "IS_VISION",
    "IS_PPO", "IS_HMO",
    "IS_SMALL_GROUP", "IS_INDIVIDUAL", "IS_SENIOR",
    "IS_PRIMARY",
    "MISSING_SSN", "MISSING_HCID",
    "IMMEDIATE_CANCEL", "CANCEL_DATE_MISMATCH",
    "FUTURE_EFFECTIVE", "NEVER_EFF_LONG_DURATION",
    "PRIMARY_CHILD", "IND_MULTI_MEMBER", "SMGRP_SINGLE_MEMBER",
    "ST_ENC", "TYP_ENC", "TP1_ENC", "MBUTY_ENC", "CR_CAT_ENC",
]

# Cancel reason code → readable category
CR_MAPPING = {
    "11": "Never_Effective",
    "08": "Non_Payment",
    "06": "Voluntary",
    "07": "Voluntary",
    "02": "Administrative",
    "01": "Administrative",
    "92": "Other",
    "05": "Other",
    "67": "Death",
    "47": "Unknown",
    "12": "Non_Payment",
    "34": "Moved_Out",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class InsurancePreprocessor:
    """
    Converts raw CSV rows (DataFrame) into a scaled numeric feature matrix.
    Call  fit_transform(df)  during training,  transform(df)  during inference.
    """

    def __init__(self):
        self.scaler   = StandardScaler()
        self.encoders: dict[str, LabelEncoder] = {}
        self.fitted   = False

    # ── public API ─────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """Training path: learn statistics AND transform."""
        df = self._engineer(df)
        df = self._encode(df, fit=True)
        X  = df[FEATURE_COLS].values.astype(float)
        X  = np.nan_to_num(X, nan=0.0)
        X  = self.scaler.fit_transform(X)
        self.fitted = True
        logger.info("Preprocessor fitted. Feature matrix: %s", X.shape)
        return X, df

    def transform(self, df: pd.DataFrame):
        """Inference path: apply already-fitted statistics."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Load or train first.")
        df = self._engineer(df)
        df = self._encode(df, fit=False)
        X  = df[FEATURE_COLS].values.astype(float)
        X  = np.nan_to_num(X, nan=0.0)
        X  = self.scaler.transform(X)
        return X, df

    def save(self, path: Path = PREPROCESSOR_PATH):
        joblib.dump(self, path)
        logger.info("Preprocessor saved → %s", path)

    @staticmethod
    def load(path: Path = PREPROCESSOR_PATH) -> "InsurancePreprocessor":
        obj = joblib.load(path)
        logger.info("Preprocessor loaded ← %s", path)
        return obj

    # ── private helpers ────────────────────────────────────────────────────────

    def _parse_date(self, series: pd.Series) -> pd.Series:
        """Parse YYYYMMDD integer column to datetime; 0 → NaT."""
        return pd.to_datetime(
            series.replace(0, np.nan).astype("Int64").astype(str)
            .replace("<NA>", np.nan),
            format="%Y%m%d", errors="coerce"
        )

    def _engineer(self, raw: pd.DataFrame) -> pd.DataFrame:
        df  = raw.copy()
        now = pd.Timestamp.now()

        # ── date columns ───────────────────────────────────────────────────────
        dob      = self._parse_date(df["MEMBIRDT"])
        eff      = self._parse_date(df["CONT EFF"])
        can      = self._parse_date(df["CONT CAN"])
        mem_eff  = self._parse_date(df["MEMEFFDT"])
        mem_can  = self._parse_date(df["MEMCANDT"])

        # ── derived numeric ────────────────────────────────────────────────────
        df["AGE"] = ((now - dob).dt.days / 365.25).fillna(0).clip(0, 120).astype(int)

        duration = (mem_can - mem_eff).dt.days
        active_dur = (now - mem_eff).dt.days
        df["CONTRACT_DURATION"] = duration.fillna(active_dur).fillna(0).astype(int)

        df["DAYS_TO_CANCEL"] = (can - eff).dt.days.fillna(0).astype(int)

        df["FAMILY_SIZE"] = df["MCNT"].fillna(1)

        # ── binary flags ───────────────────────────────────────────────────────
        df["IS_CANCELLED"]   = (df["STS"] != 0).astype(int)
        df["IS_MEDICAL"]     = (df["TYP"] == "MED").astype(int)
        df["IS_DENTAL"]      = (df["TYP"] == "DEN").astype(int)
        df["IS_VISION"]      = (df["TYP"] == "VIS").astype(int)
        df["IS_PPO"]         = (df["TP1"] == "PPO").astype(int)
        df["IS_HMO"]         = (df["TP1"] == "HMO").astype(int)
        df["IS_SMALL_GROUP"] = (df["MBUTY"] == "SMGRP").astype(int)
        df["IS_INDIVIDUAL"]  = (df["MBUTY"] == "IND").astype(int)
        df["IS_SENIOR"]      = (df["MBUTY"] == "SEN").astype(int)
        df["IS_PRIMARY"]     = df["MCDE"].isin([10, 20]).astype(int)

        # ── data quality flags ─────────────────────────────────────────────────
        df["MISSING_SSN"]  = df["SSN"].isna().astype(int)
        df["MISSING_HCID"] = df["HCID"].isna().astype(int)

        # ── anomaly / business-rule flags ──────────────────────────────────────
        df["IMMEDIATE_CANCEL"] = (
            eff.notna() & can.notna() & (eff == can)
        ).astype(int)

        df["CANCEL_DATE_MISMATCH"] = (
            can.notna() & mem_can.notna() & (can != mem_can)
        ).astype(int)

        df["FUTURE_EFFECTIVE"] = (eff > now).astype(int)

        cr_str = df["CR"].astype(str).str.strip().str.zfill(2)
        df["NEVER_EFF_LONG_DURATION"] = (
            (cr_str == "11") & (df["CONTRACT_DURATION"] > 30)
        ).astype(int)

        df["PRIMARY_CHILD"]       = (df["MCDE"].isin([10, 20]) & (df["AGE"] < 18)).astype(int)
        df["IND_MULTI_MEMBER"]    = ((df["MBUTY"] == "IND")   & (df["MCNT"] > 1)).astype(int)
        df["SMGRP_SINGLE_MEMBER"] = ((df["MBUTY"] == "SMGRP") & (df["MCNT"] == 1)).astype(int)

        # ── cancel reason category ─────────────────────────────────────────────
        df["CR_CAT"] = cr_str.map(CR_MAPPING).fillna("Unknown")

        return df

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label-encode each categorical column."""
        for col in CAT_COLS:
            enc_col = f"{col}_ENC"
            series  = df[col].fillna("Unknown").astype(str)
            if fit:
                le = LabelEncoder()
                le.fit(list(series.unique()) + ["Unknown"])
                self.encoders[col] = le
            le = self.encoders.get(col)
            if le is None:
                df[enc_col] = 0
                continue
            safe = series.apply(
                lambda v: v if v in le.classes_ else "Unknown"
            )
            df[enc_col] = le.transform(safe)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class InsuranceAnomalyDetector:
    """
    Thin wrapper around IsolationForest.
    Scores are normalised to [0, 1] where 1 = most anomalous.
    """

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            max_samples="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.trained = False

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray):
        logger.info("Training IsolationForest on %s …", X.shape)
        self.model.fit(X)
        self.trained = True
        logger.info("Training complete.")
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Returns dict with:
          labels  : +1 = normal, -1 = anomaly
          scores  : raw decision_function scores (higher = more normal)
          norm_scores: [0, 1] anomaly score (higher = more anomalous)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Load or train first.")
        labels = self.model.predict(X)                       # +1 / -1
        scores = self.model.decision_function(X)             # raw
        # Normalise to [0, 1] anomaly score
        mn, mx = scores.min(), scores.max()
        if mx > mn:
            norm = 1.0 - (scores - mn) / (mx - mn)          # invert: low score → high anomaly
        else:
            norm = np.zeros_like(scores, dtype=float)
        return {"labels": labels, "raw_scores": scores, "anomaly_scores": norm}

    def save(self, path: Path = MODEL_PATH):
        joblib.dump(self, path)
        logger.info("Model saved → %s", path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "InsuranceAnomalyDetector":
        obj = joblib.load(path)
        logger.info("Model loaded ← %s", path)
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════════

def explain_row(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("MISSING_SSN", 0):
        reasons.append("Missing SSN — critical identifier absent")
    if row.get("MISSING_HCID", 0):
        reasons.append("Missing HCID — Health Care ID absent")
    if row.get("PRIMARY_CHILD", 0):
        reasons.append(f"Primary subscriber is under 18 (age {int(row.get('AGE', 0))})")
    if row.get("IND_MULTI_MEMBER", 0):
        reasons.append(f"Individual plan with multiple members (MCNT={int(row.get('MCNT', 0))})")
    if row.get("SMGRP_SINGLE_MEMBER", 0):
        reasons.append("Small Group plan with only 1 member")
    if row.get("IMMEDIATE_CANCEL", 0):
        reasons.append("Cancelled on the same day as effective date")
    if row.get("CANCEL_DATE_MISMATCH", 0):
        reasons.append("Contract cancel date ≠ member cancel date")
    if row.get("FUTURE_EFFECTIVE", 0):
        reasons.append("Effective date is in the future")
    if row.get("NEVER_EFF_LONG_DURATION", 0):
        reasons.append(f"'Never Effective' cancellation but contract lasted {int(row.get('CONTRACT_DURATION', 0))} days")
    age = row.get("AGE", 0)
    if age > 100:
        reasons.append(f"Age exceeds 100 ({age})")
    if row.get("DAYS_TO_CANCEL", 0) > 0 and row.get("DAYS_TO_CANCEL", 0) < 30:
        reasons.append(f"Cancelled within {int(row.get('DAYS_TO_CANCEL', 0))} days of enrollment")
    if row.get("FAMILY_SIZE", 1) > 15:
        reasons.append(f"Unusually large family size (MCNT={int(row.get('FAMILY_SIZE', 0))})")
    if not reasons:
        reasons.append("Statistical outlier — deviates from learned normal pattern")
    return reasons


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE  (used by train script and inference service)
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyPipeline:
    """Combines preprocessor + detector. Single object for inference."""

    def __init__(self):
        self.preprocessor = InsurancePreprocessor()
        self.detector     = InsuranceAnomalyDetector()

    # ── training ───────────────────────────────────────────────────────────────

    def train(self, csv_path: str):
        logger.info("Loading %s …", csv_path)
        df   = pd.read_csv(csv_path, low_memory=False)
        X, _ = self.preprocessor.fit_transform(df)
        self.detector.fit(X)
        return self

    def save(self):
        self.preprocessor.save()
        self.detector.save()
        logger.info("Pipeline saved.")

    # ── inference ──────────────────────────────────────────────────────────────

    @staticmethod
    def load() -> "AnomalyPipeline":
        p = AnomalyPipeline()
        p.preprocessor = InsurancePreprocessor.load()
        p.detector     = InsuranceAnomalyDetector.load()
        return p

    def predict_df(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Accept a DataFrame of raw member records.
        Returns a list of prediction dicts — one per row.
        """
        X, enriched = self.preprocessor.transform(df)
        result      = self.detector.predict(X)

        labels  = result["labels"]
        a_scores = result["anomaly_scores"]
        raw      = result["raw_scores"]

        output = []
        for i, row in enumerate(enriched.itertuples(index=False)):
            row_dict  = row._asdict()
            is_anomaly = bool(labels[i] == -1)
            reasons    = explain_row(pd.Series(row_dict)) if is_anomaly else []
            output.append({
                "hcid":           str(row_dict.get("HCID", "unknown")),
                "is_anomaly":     is_anomaly,
                "anomaly_status": "ANOMALY" if is_anomaly else "NORMAL",
                "anomaly_score":  round(float(a_scores[i]), 6),
                "raw_score":      round(float(raw[i]), 6),
                "anomaly_reasons": reasons,
            })
        return output
