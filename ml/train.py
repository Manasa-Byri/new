"""
Training Script for Anomaly Detection Models
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from ml.config import CSV_FILE, RESULTS_DIR, MODELS_DIR
from ml.data_preprocessing import InsuranceDataPreprocessor
from ml.anomaly_detector import AnomalyDetector, AnomalyExplainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models():
    """
    Main training pipeline
    """
    logger.info("="*80)
    logger.info("ANOMALY DETECTION MODEL TRAINING")
    logger.info("="*80)
    
    # Step 1: Load and preprocess data
    logger.info("\n[Step 1] Loading and preprocessing data...")
    preprocessor = InsuranceDataPreprocessor()
    X, df = preprocessor.preprocess(str(CSV_FILE), fit=True)
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {preprocessor.get_feature_importance_names()}")
    
    # Step 2: Train anomaly detection models
    logger.info("\n[Step 2] Training anomaly detection models...")
    detector = AnomalyDetector()
    models = detector.train_all_models(X)
    
    logger.info(f"Trained models: {list(models.keys())}")
    
    # Step 3: Evaluate models
    logger.info("\n[Step 3] Evaluating models...")
    evaluation_results = {}
    
    for model_name in models.keys():
        logger.info(f"\nEvaluating {model_name}...")
        
        # Get predictions
        predictions = detector.predict(X, model_name)
        anomalies_df = detector.analyze_anomalies(X, df, model_name)
        
        # Get summary
        summary = detector.get_anomaly_summary(anomalies_df)
        
        evaluation_results[model_name] = {
            'num_anomalies': len(anomalies_df),
            'anomaly_rate': round((len(anomalies_df) / len(df)) * 100, 2),
            'summary': summary
        }
        
        logger.info(f"  Anomalies detected: {len(anomalies_df)} ({evaluation_results[model_name]['anomaly_rate']}%)")
    
    # Step 4: Ensemble prediction
    logger.info("\n[Step 4] Running ensemble prediction...")
    ensemble_preds, model_votes = detector.predict_ensemble(X, threshold=0.5)
    ensemble_anomalies = df[ensemble_preds == -1].copy()
    
    logger.info(f"Ensemble detected {len(ensemble_anomalies)} anomalies")
    
    # Step 5: Add explanations
    logger.info("\n[Step 5] Generating anomaly explanations...")
    ensemble_anomalies = AnomalyExplainer.add_explanations(ensemble_anomalies)
    
    # Step 6: Save results
    logger.info("\n[Step 6] Saving results...")
    
    # Save models
    detector.save_models(MODELS_DIR)
    
    # Save preprocessor
    import joblib
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    # Save anomalies
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    anomalies_file = RESULTS_DIR / f"anomalies_{timestamp}.csv"
    ensemble_anomalies.to_csv(anomalies_file, index=False)
    logger.info(f"Saved anomalies to {anomalies_file}")
    
    # Save evaluation report
    report_file = RESULTS_DIR / f"evaluation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("ANOMALY DETECTION EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {CSV_FILE}\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Features: {X.shape[1]}\n\n")
        
        f.write("MODEL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        for model_name, results in evaluation_results.items():
            f.write(f"\n{model_name.upper()}\n")
            f.write(f"  Anomalies: {results['num_anomalies']}\n")
            f.write(f"  Anomaly Rate: {results['anomaly_rate']}%\n")
        
        f.write(f"\n\nENSEMBLE PREDICTION\n")
        f.write("-"*80 + "\n")
        f.write(f"Anomalies: {len(ensemble_anomalies)}\n")
        f.write(f"Anomaly Rate: {round((len(ensemble_anomalies)/len(df))*100, 2)}%\n")
        
        f.write(f"\n\nANOMALY SUMMARY\n")
        f.write("-"*80 + "\n")
        summary = detector.get_anomaly_summary(ensemble_anomalies)
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved evaluation report to {report_file}")
    
    # Step 7: Display sample anomalies
    logger.info("\n[Step 7] Sample Anomalies:")
    logger.info("-"*80)
    
    sample_anomalies = ensemble_anomalies.head(10)
    for idx, row in sample_anomalies.iterrows():
        logger.info(f"\nAnomaly #{idx}")
        logger.info(f"  HCID: {row.get('HCID', 'N/A')}")
        logger.info(f"  State: {row.get('ST', 'N/A')}")
        logger.info(f"  Age: {row.get('AGE', 'N/A')}")
        logger.info(f"  Coverage: {row.get('TYP', 'N/A')}")
        logger.info(f"  Status: {'Active' if row.get('STS') == 0 else 'Inactive'}")
        logger.info(f"  Reasons: {', '.join(row.get('anomaly_reasons', []))}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nModels saved to: {MODELS_DIR}")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    
    return {
        'detector': detector,
        'preprocessor': preprocessor,
        'evaluation_results': evaluation_results,
        'ensemble_anomalies': ensemble_anomalies
    }


if __name__ == "__main__":
    try:
        results = train_models()
        logger.info("\n✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}", exc_info=True)
        sys.exit(1)
