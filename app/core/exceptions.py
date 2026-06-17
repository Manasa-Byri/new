"""
Domain exceptions for the AI Insight Agent.

All custom exceptions inherit from AnomalyServiceError so callers can catch
the base type when they do not care about the specific sub-class.
"""
from __future__ import annotations


class AnomalyServiceError(Exception):
    """Base class for all AI Insight Agent exceptions."""


# ── model ─────────────────────────────────────────────────────────────────────

class ModelNotFoundError(AnomalyServiceError):
    """Raised when an expected model artifact (.joblib) does not exist."""


class ModelLoadError(AnomalyServiceError):
    """Raised when a model artifact cannot be deserialised."""


class ModelNotTrainedError(AnomalyServiceError):
    """Raised when score() is called before fit() or load()."""


# ── data ──────────────────────────────────────────────────────────────────────

class DataLoaderError(AnomalyServiceError):
    """Base class for data-loading failures."""


class PostgresUnavailableError(DataLoaderError):
    """Raised when the PostgreSQL database cannot be reached."""


class MongoUnavailableError(DataLoaderError):
    """Raised when the MongoDB instance cannot be reached."""


class CSVNotFoundError(DataLoaderError):
    """Raised when the configured CSV file path does not exist."""


# ── input validation ──────────────────────────────────────────────────────────

class InvalidInputError(AnomalyServiceError):
    """
    Raised when the caller supplies a payload that fails validation.

    Attributes
    ----------
    detail : list[dict]
        Pydantic-style list of error dicts with ``loc``, ``msg``, ``type``.
    """

    def __init__(self, detail: list[dict] | str) -> None:
        if isinstance(detail, str):
            detail = [{"msg": detail}]
        self.detail = detail
        super().__init__(str(detail))
