"""
Settings for the AI Insight Agent anomaly-detection service.
All values are driven by environment variables — NO hardcoded credentials.

Copy .env.example → .env and fill in real values before running.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── identity ──────────────────────────────────────────────────────────────
    APP_NAME: str    = "ai-insight-agent"
    APP_VERSION: str = "1.0.0"
    ENV: str         = Field("development", description="development | staging | production")

    # ── logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field("INFO", description="DEBUG | INFO | WARNING | ERROR")

    # ── model artifact ────────────────────────────────────────────────────────
    MODEL_DIR: Path = Field(
        Path(__file__).resolve().parent.parent.parent / "ml" / "models",
        description="Directory containing saved model + preprocessor .joblib files",
    )
    MODEL_FILENAME: str       = "v1_isolation_forest.joblib"
    PREPROCESSOR_FILENAME: str = "v1_preprocessor.joblib"
    RANDOM_STATE: int = 42

    # ── anomaly detection ─────────────────────────────────────────────────────
    # TODO: confirm contamination rate with SMEs — current assumption is 5 %
    CONTAMINATION: float = Field(0.05, ge=0.001, le=0.5)
    N_ESTIMATORS: int    = Field(200,  ge=50,     le=2000)
    MAX_SAMPLES: str     = "auto"     # "auto" | integer

    # ── severity thresholds ───────────────────────────────────────────────────
    # anomaly_score is normalised [0, 1]; higher = more anomalous.
    # TODO: calibrate thresholds against confirmed incidents in production.
    SEVERITY_CRITICAL_THRESHOLD: float = Field(0.80, ge=0.0, le=1.0)
    SEVERITY_HIGH_THRESHOLD:     float = Field(0.60, ge=0.0, le=1.0)
    SEVERITY_MEDIUM_THRESHOLD:   float = Field(0.40, ge=0.0, le=1.0)
    # scores below MEDIUM_THRESHOLD → "LOW" (informational, not alerted)

    # ── PostgreSQL (read-only) ────────────────────────────────────────────────
    POSTGRES_HOST: Optional[str]     = None
    POSTGRES_PORT: int               = 5432
    POSTGRES_DB: Optional[str]       = None
    POSTGRES_USER: Optional[str]     = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_CONNECT_TIMEOUT: int    = Field(10,  description="seconds")
    POSTGRES_QUERY_TIMEOUT_MS: int   = Field(30000, description="milliseconds")
    POSTGRES_MAX_RETRIES: int        = 3
    POSTGRES_RETRY_BACKOFF_S: float  = 1.0

    # ── MongoDB (read-only) ───────────────────────────────────────────────────
    MONGODB_URL: Optional[str]        = None
    MONGODB_DB: str                   = "file_tracking"
    MONGODB_CONNECT_TIMEOUT_MS: int   = 5000
    MONGODB_SOCKET_TIMEOUT_MS: int    = 30000
    MONGODB_MAX_RETRIES: int          = 3

    # ── CSV source ────────────────────────────────────────────────────────────
    CSV_PATH: Optional[Path] = Field(
        None,
        description="Absolute path to member CSV file. Auto-detected if absent.",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_WORKERS: int = 1
    API_RATE_LIMIT: int = 100  # requests per minute per IP
    API_RATE_LIMIT_PERIOD: int = 60

    # ── legacy fields (kept for backward-compat with existing app/ modules) ──
    DATABASE_URL: Optional[str]    = None
    DATABASE_POOL_SIZE: int        = 5
    DATABASE_MAX_OVERFLOW: int     = 10
    AWS_REGION: str                = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str]     = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    CLOUDWATCH_LOG_GROUP: Optional[str]  = None
    REDIS_URL: Optional[str]       = None
    CACHE_TTL: int                 = 300
    CORS_ORIGINS: list             = ["*"]
    CORS_ALLOW_CREDENTIALS: bool   = True
    CORS_ALLOW_METHODS: list       = ["*"]
    CORS_ALLOW_HEADERS: list       = ["*"]
    DEBUG: bool                    = False

    @field_validator("ENV")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENV must be one of {allowed}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# ── singleton ─────────────────────────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
