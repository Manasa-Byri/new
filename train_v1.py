"""
train_v1.py — Train and save the v1 Isolation Forest model.

Usage
-----
    python train_v1.py [--csv PATH] [--model-dir DIR]

Artifacts saved
---------------
    <model_dir>/v1_isolation_forest.joblib
    <model_dir>/v1_preprocessor.joblib

Design: this script is the ONLY place where model fitting happens.
Serving code (app/model/scorer.py) only calls scorer.load() — never fit().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package root is importable when run directly.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.settings import get_settings
from app.core.logging import configure_logging, get_logger
from app.data.loaders import CSVLoader
from app.model.scorer import Scorer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v1 Isolation Forest anomaly model")
    p.add_argument("--csv",       type=Path, default=None, help="Path to member CSV file")
    p.add_argument("--model-dir", type=Path, default=None, help="Directory to save model artifacts")
    return p.parse_args()


def main() -> None:
    configure_logging("INFO")
    args     = parse_args()
    settings = get_settings()

    # ── load data ─────────────────────────────────────────────────────────────
    loader = CSVLoader()
    if args.csv:
        import os
        os.environ["CSV_PATH"] = str(args.csv.resolve())
        # Re-initialise settings with the new env var
        from app.config import settings as settings_module
        settings_module._settings = None  # type: ignore[attr-defined]

    df = loader.load()
    logger.info("Training data loaded", extra={"rows": len(df), "cols": len(df.columns)})

    # ── train ─────────────────────────────────────────────────────────────────
    scorer = Scorer()
    scorer.fit(df)

    # ── save ──────────────────────────────────────────────────────────────────
    model_dir = args.model_dir or settings.MODEL_DIR
    saved_to  = scorer.save(model_dir)
    logger.info("Training complete", extra={"model_dir": str(saved_to)})

    # ── quick sanity-check score ──────────────────────────────────────────────
    sample  = df.sample(min(5, len(df)), random_state=42)
    results = scorer.score(sample)
    print("\n=== Sample predictions ===")
    for r in results:
        print(
            f"  hcid={r.hcid!r:15s}  "
            f"is_anomaly={r.is_anomaly!s:5s}  "
            f"score={r.score:.4f}  "
            f"severity={r.severity.value:8s}  "
            f"reasons={r.anomaly_reasons}"
        )
    print()


if __name__ == "__main__":
    main()
