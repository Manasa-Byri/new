"""
Training script — runs once to build and save the anomaly model.
Usage:
    .\venv\Scripts\python.exe ml/train_insurance.py
"""
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.insurance_anomaly import AnomalyPipeline, MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

CSV_PATH = Path(__file__).resolve().parent.parent / "USRW.NONX.IYM551ND.MEMBER.SWEEP.G2262V.csv"


def main():
    logger.info("=" * 60)
    logger.info("INSURANCE ANOMALY MODEL — TRAINING")
    logger.info("=" * 60)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    # ── Step 1: build pipeline ────────────────────────────────────────────────
    pipeline = AnomalyPipeline()

    # ── Step 2: train ─────────────────────────────────────────────────────────
    pipeline.train(str(CSV_PATH))

    # ── Step 3: quick evaluation on training data ─────────────────────────────
    import pandas as pd
    df = pd.read_csv(CSV_PATH, low_memory=False)
    predictions = pipeline.predict_df(df)

    anomalies  = [p for p in predictions if p["is_anomaly"]]
    normal     = [p for p in predictions if not p["is_anomaly"]]
    avg_score  = sum(p["anomaly_score"] for p in anomalies) / len(anomalies) if anomalies else 0

    logger.info("-" * 60)
    logger.info("Total records   : %d", len(predictions))
    logger.info("Anomalies found : %d  (%.1f%%)", len(anomalies), len(anomalies) / len(predictions) * 100)
    logger.info("Normal records  : %d", len(normal))
    logger.info("Avg anomaly score (anomalies only): %.4f", avg_score)

    # sample
    logger.info("\nSample anomalies:")
    for p in anomalies[:5]:
        logger.info("  HCID: %-15s  score: %.4f  reasons: %s",
                    p["hcid"], p["anomaly_score"], "; ".join(p["anomaly_reasons"][:2]))

    # ── Step 4: save ──────────────────────────────────────────────────────────
    pipeline.save()

    logger.info("-" * 60)
    logger.info("Model saved     → %s", MODEL_PATH)
    logger.info("Preprocessor    → %s", PREPROCESSOR_PATH)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
