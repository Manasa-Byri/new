"""
Unit tests for app/data/loaders.py — fault-tolerant data loaders.

All tests are hermetic: no real DB/Mongo/file access.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.core.exceptions import CSVNotFoundError, MongoUnavailableError, PostgresUnavailableError
from app.data.loaders import CSVLoader, MongoLoader, PostgresLoader


# ── CSVLoader ──────────────────────────────────────────────────────────────────


class TestCSVLoader:
    def test_load_returns_dataframe(self, tmp_path: Path):
        csv = tmp_path / "members.csv"
        csv.write_text("HCID,MCNT,MCDE\nABC001,2,20\nABC002,1,10\n")
        with patch("app.data.loaders.get_settings") as mock_settings:
            mock_settings.return_value.CSV_PATH = csv
            loader = CSVLoader()
            df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_load_drops_rx_column(self, tmp_path: Path):
        csv = tmp_path / "members.csv"
        csv.write_text("HCID,MCNT,RX\nABC001,2,1.0\n")
        with patch("app.data.loaders.get_settings") as mock_settings:
            mock_settings.return_value.CSV_PATH = csv
            loader = CSVLoader()
            df = loader.load()
        assert "RX" not in df.columns

    def test_missing_csv_raises(self, tmp_path: Path):
        nonexistent = tmp_path / "nonexistent.csv"
        with patch("app.data.loaders.get_settings") as mock_settings:
            s = MagicMock()
            s.CSV_PATH = nonexistent   # points at a path that doesn't exist
            mock_settings.return_value = s
            loader = CSVLoader()
            # Patch the auto-detect candidates list to also be empty
            with patch.object(loader, "_resolve_path", side_effect=CSVNotFoundError("not found")):
                with pytest.raises(CSVNotFoundError):
                    loader.load()

    def test_iter_chunks_yields_dataframes(self, tmp_path: Path):
        rows = "\n".join(f"MBR{i:03d},{i % 5 + 1},20" for i in range(20))
        csv = tmp_path / "members.csv"
        csv.write_text("HCID,MCNT,MCDE\n" + rows + "\n")
        with patch("app.data.loaders.get_settings") as mock_settings:
            mock_settings.return_value.CSV_PATH = csv
            loader  = CSVLoader()
            chunks  = list(loader.iter_chunks(chunksize=7))
        total = sum(len(c) for c in chunks)
        assert total == 20
        assert all(isinstance(c, pd.DataFrame) for c in chunks)


# ── PostgresLoader ─────────────────────────────────────────────────────────────


class TestPostgresLoader:
    def test_load_enrollments_returns_empty_when_unconfigured(self):
        """If POSTGRES_HOST is unset, return empty DataFrame (not raise)."""
        with patch("app.data.loaders.get_settings") as mock_settings:
            s = MagicMock()
            s.POSTGRES_HOST     = None
            s.POSTGRES_DB       = None
            s.POSTGRES_USER     = None
            s.POSTGRES_PASSWORD = None
            s.POSTGRES_MAX_RETRIES     = 1
            s.POSTGRES_RETRY_BACKOFF_S = 0.0
            mock_settings.return_value = s
            loader = PostgresLoader()
            df = loader.load_enrollments()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_enrollments_returns_empty_on_connection_failure(self):
        """Connection error is caught and empty DataFrame is returned."""
        with patch("app.data.loaders.get_settings") as mock_settings:
            s = MagicMock()
            s.POSTGRES_HOST     = "127.0.0.1"
            s.POSTGRES_PORT     = 5432
            s.POSTGRES_DB       = "testdb"
            s.POSTGRES_USER     = "postgres"
            s.POSTGRES_PASSWORD = "secret"
            s.POSTGRES_CONNECT_TIMEOUT   = 1
            s.POSTGRES_QUERY_TIMEOUT_MS  = 1000
            s.POSTGRES_MAX_RETRIES       = 0
            s.POSTGRES_RETRY_BACKOFF_S   = 0.0
            mock_settings.return_value = s
            loader = PostgresLoader()
            df = loader.load_enrollments()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ── MongoLoader ────────────────────────────────────────────────────────────────


class TestMongoLoader:
    def test_load_audit_stats_returns_empty_when_unconfigured(self):
        with patch("app.data.loaders.get_settings") as mock_settings:
            s = MagicMock()
            s.MONGODB_URL = None
            s.MONGODB_MAX_RETRIES = 1
            mock_settings.return_value = s
            loader = MongoLoader()
            stats = loader.load_audit_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0

    def test_load_file_stats_returns_empty_when_unconfigured(self):
        with patch("app.data.loaders.get_settings") as mock_settings:
            s = MagicMock()
            s.MONGODB_URL = None
            s.MONGODB_MAX_RETRIES = 1
            mock_settings.return_value = s
            loader = MongoLoader()
            stats = loader.load_file_processing_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0
