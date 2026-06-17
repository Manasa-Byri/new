"""
Read-only data loaders for the AI Insight Agent.

Three sources are supported:
  - PostgreSQL  (canonical_enrollments and related tables)
  - MongoDB     (audit_logs, ingestion_files)
  - CSV         (local flat-file source for offline / dev use)

All loaders are:
  * read-only — no INSERT / UPDATE / DELETE is issued anywhere.
  * fault-tolerant — connection errors raise typed exceptions from app.core.exceptions,
    NOT raw driver exceptions; callers can handle each source independently.
  * retried — transient network errors are retried with exponential back-off.
  * timeout-bounded — both connect and query timeouts are configurable via Settings.

Sensitive connection strings come ONLY from environment variables / Settings.
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from app.config.settings import get_settings
from app.core.exceptions import (
    CSVNotFoundError,
    DataLoaderError,
    MongoUnavailableError,
    PostgresUnavailableError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────


def _retry(
    fn,
    *,
    max_retries: int,
    backoff_s: float,
    exc_type: type[DataLoaderError],
    label: str,
):
    """Call fn(); retry up to max_retries times on any exception."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):  # +1 for the initial attempt
        try:
            return fn()
        except exc_type:
            raise  # already typed; don't retry — already a final error
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt <= max_retries:
                wait = backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    f"{label} attempt {attempt} failed, retrying in {wait:.1f}s",
                    extra={"error": str(exc)},
                )
                time.sleep(wait)
    raise exc_type(f"{label} failed after {max_retries + 1} attempts: {last_exc}") from last_exc


# ── PostgreSQL loader ─────────────────────────────────────────────────────────


class PostgresLoader:
    """
    Read-only access to the enrollment PostgreSQL database.

    All queries use parameterised statements and read-committed isolation.
    No write operations are available by design.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def _build_dsn(self) -> str:
        s = self._settings
        if not all([s.POSTGRES_HOST, s.POSTGRES_DB, s.POSTGRES_USER, s.POSTGRES_PASSWORD]):
            raise PostgresUnavailableError(
                "PostgreSQL connection is not configured. "
                "Set POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD in env."
            )
        # NOTE: password is NOT logged here
        return (
            f"host={s.POSTGRES_HOST} port={s.POSTGRES_PORT} "
            f"dbname={s.POSTGRES_DB} user={s.POSTGRES_USER} "
            f"password={s.POSTGRES_PASSWORD} "
            f"connect_timeout={s.POSTGRES_CONNECT_TIMEOUT} "
            f"options=-c statement_timeout={s.POSTGRES_QUERY_TIMEOUT_MS}"
        )

    def _connect(self):
        try:
            import psycopg2  # type: ignore
            from psycopg2.extras import RealDictCursor  # type: ignore

            conn = psycopg2.connect(self._build_dsn(), cursor_factory=RealDictCursor)
            conn.set_session(readonly=True, autocommit=True)
            return conn
        except ImportError as exc:
            raise PostgresUnavailableError("psycopg2 is not installed") from exc
        except Exception as exc:
            raise PostgresUnavailableError(f"Cannot connect to PostgreSQL: {exc}") from exc

    def load_enrollments(self) -> pd.DataFrame:
        """
        Load the canonical_enrollments table for anomaly scoring.

        Returns an empty DataFrame on connection failure so the caller can
        decide whether to fall back to the CSV source.
        """
        s = self._settings

        def _fetch() -> pd.DataFrame:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    # TODO: confirm complete column list with data engineering team.
                    cur.execute(
                        """
                        SELECT
                            member_id,
                            validation_status,
                            hold_flag,
                            state_code,
                            file_name
                        FROM canonical_enrollments
                        LIMIT 100000
                        """
                    )
                    rows = cur.fetchall()
                    return pd.DataFrame([dict(r) for r in rows])
            finally:
                conn.close()

        try:
            df = _retry(
                _fetch,
                max_retries=s.POSTGRES_MAX_RETRIES,
                backoff_s=s.POSTGRES_RETRY_BACKOFF_S,
                exc_type=PostgresUnavailableError,
                label="PostgreSQL.load_enrollments",
            )
            logger.info("PostgreSQL enrollments loaded", extra={"rows": len(df)})
            return df
        except PostgresUnavailableError as exc:
            logger.warning("PostgreSQL unavailable — skipping enrollment load", extra={"error": str(exc)})
            return pd.DataFrame()


# ── MongoDB loader ────────────────────────────────────────────────────────────


class MongoLoader:
    """
    Read-only access to the MongoDB file-tracking and audit databases.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def _get_client(self):
        s = self._settings
        if not s.MONGODB_URL:
            raise MongoUnavailableError(
                "MongoDB connection is not configured. Set MONGODB_URL in env."
            )
        try:
            from pymongo import MongoClient  # type: ignore
            from pymongo.errors import ConnectionFailure  # type: ignore

            client = MongoClient(
                s.MONGODB_URL,
                connectTimeoutMS=s.MONGODB_CONNECT_TIMEOUT_MS,
                socketTimeoutMS=s.MONGODB_SOCKET_TIMEOUT_MS,
                serverSelectionTimeoutMS=s.MONGODB_CONNECT_TIMEOUT_MS,
            )
            # Force a connection check
            client.admin.command("ping")
            return client
        except ImportError as exc:
            raise MongoUnavailableError("pymongo is not installed") from exc
        except Exception as exc:
            raise MongoUnavailableError(f"Cannot connect to MongoDB: {exc}") from exc

    def load_audit_stats(self) -> dict[str, Any]:
        """
        Return aggregated API performance metrics from audit_logs.

        Returns an empty dict when MongoDB is unavailable.
        """
        s = self._settings

        def _fetch() -> dict[str, Any]:
            client = self._get_client()
            try:
                db  = client[s.MONGODB_DB]
                col = db["audit_logs"]
                pipeline = [
                    {"$group": {
                        "_id":                    None,
                        "total_requests":         {"$sum": 1},
                        "avg_response_time_ms":   {"$avg": "$response_time_ms"},
                        "max_response_time_ms":   {"$max": "$response_time_ms"},
                        "p95_response_time_ms":   {"$percentile": {
                            "input": "$response_time_ms",
                            "p": [0.95],
                            "method": "approximate",
                        }},
                        "cache_hit_count":        {"$sum": {"$cond": [{"$eq": ["$cache_hit_indicator", True]}, 1, 0]}},
                    }},
                ]
                result = list(col.aggregate(pipeline))
                return result[0] if result else {}
            finally:
                client.close()

        try:
            stats = _retry(
                _fetch,
                max_retries=s.MONGODB_MAX_RETRIES,
                backoff_s=1.0,
                exc_type=MongoUnavailableError,
                label="MongoDB.load_audit_stats",
            )
            stats.pop("_id", None)
            return stats
        except MongoUnavailableError as exc:
            logger.warning("MongoDB unavailable — skipping audit stats", extra={"error": str(exc)})
            return {}

    def load_file_processing_stats(self) -> dict[str, Any]:
        """
        Return file-processing success/failure counts from ingestion_files.

        Returns an empty dict when MongoDB is unavailable.
        """
        s = self._settings

        def _fetch() -> dict[str, Any]:
            client = self._get_client()
            try:
                db  = client[s.MONGODB_DB]
                col = db["ingestion_files"]
                pipeline = [
                    {"$group": {
                        "_id":              None,
                        "total_files":      {"$sum": 1},
                        "total_errors":     {"$sum": "$error_count"},
                        "total_records":    {"$sum": "$total_records"},
                        "members_failed":   {"$sum": "$members_failed"},
                        "members_succeeded":{"$sum": "$members_succeeded"},
                    }},
                ]
                result = list(col.aggregate(pipeline))
                return result[0] if result else {}
            finally:
                client.close()

        try:
            stats = _retry(
                _fetch,
                max_retries=s.MONGODB_MAX_RETRIES,
                backoff_s=1.0,
                exc_type=MongoUnavailableError,
                label="MongoDB.load_file_processing_stats",
            )
            stats.pop("_id", None)
            return stats
        except MongoUnavailableError as exc:
            logger.warning("MongoDB unavailable — skipping file stats", extra={"error": str(exc)})
            return {}


# ── CSV loader ────────────────────────────────────────────────────────────────


class CSVLoader:
    """
    Load member data from the local flat-file CSV source.

    Intended for offline development and as a fallback when the database is
    unreachable.  Uses chunked reading to avoid exhausting memory on large files.
    """

    # Columns we know carry no signal for anomaly detection
    _DROP_COLS = {"RX", "Unnamed: 37"}

    def __init__(self) -> None:
        self._settings = get_settings()

    def _resolve_path(self) -> Path:
        s = self._settings
        if s.CSV_PATH and s.CSV_PATH.exists():
            return s.CSV_PATH
        # Auto-detect: walk up from this file's location looking for the CSV
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "USRW.NONX.IYM551ND.MEMBER.SWEEP.G2262V.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise CSVNotFoundError(
            "CSV file not found. Set CSV_PATH in .env or place the file at the project root."
        )

    def load(self, *, chunksize: Optional[int] = None) -> pd.DataFrame:
        """
        Load the full CSV into a DataFrame.

        Parameters
        ----------
        chunksize : int, optional
            If supplied, load in chunks of this many rows (lower memory usage).
            The returned DataFrame is still the fully concatenated result.

        Returns
        -------
        pd.DataFrame
            Cleaned member data; empty / all-null columns are dropped.
        """
        path = self._resolve_path()
        logger.info("Loading CSV", extra={"path": str(path)})
        try:
            if chunksize:
                chunks = []
                for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            raise CSVNotFoundError(f"Failed to read CSV at {path}: {exc}") from exc

        # Drop known-empty columns
        df.drop(columns=[c for c in self._DROP_COLS if c in df.columns], inplace=True)
        logger.info("CSV loaded", extra={"rows": len(df), "cols": len(df.columns)})
        return df

    def iter_chunks(self, chunksize: int = 5000) -> Iterator[pd.DataFrame]:
        """Yield chunks of the CSV for large-scale batch scoring."""
        path = self._resolve_path()
        try:
            for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
                chunk.drop(columns=[c for c in self._DROP_COLS if c in chunk.columns], inplace=True)
                yield chunk
        except Exception as exc:
            raise CSVNotFoundError(f"Failed to stream CSV at {path}: {exc}") from exc
