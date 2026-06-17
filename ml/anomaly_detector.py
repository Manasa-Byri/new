"""
Anomaly Detection Models for Insurance Data
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple

from ml.config import MODELS_CONFIG, CONTAMINATION, MODELS_DIR

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Multi-model anomaly detection system
    """
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.trained = False
        
    def train_isolation_forest(self, X: np.ndarray) -> IsolationForest:
        """
        Train Isolation Forest model
        Best for: High-dimensional data, fast training
        """
        logger.info("Training Isolation Forest...")
        config = MODELS_CONFIG['isolation_forest']
        
        model = IsolationForest(**config)
        model.fit(X)
        
        logger.info("Isolation Forest training complete")
        return model
    
    def train_local_outlier_factor(self, X: np.ndarray) -> LocalOutlierFactor:
        """
        Train Local Outlier Factor model
        Best for: Density-based anomalies
        """
        logger.info("Training Local Outlier Factor...")
        config = MODELS_CONFIG['local_outlier_factor']
        
        model = LocalOutlierFactor(**config)
        model.fit(X)
        
        logger.info("Local Outlier Factor training complete")
        return model
    
    def train_one_class_svm(self, X: np.ndarray) -> OneClassSVM:
        """
        Train One-Class SVM model
        Best for: Non-linear boundaries
        """
        logger.info("Training One-Class SVM...")
        config = MODELS_CONFIG['one_class_svm']
        
        model = OneClassSVM(**config)
        model.fit(X)
        
        logger.info("One-Class SVM training complete")
        return model
    
    def train_elliptic_envelope(self, X: np.ndarray) -> EllipticEnvelope:
        """
        Train Elliptic Envelope model
        Best for: Gaussian distributed data
        """
        logger.info("Training Elliptic Envelope...")
        
        model = EllipticEnvelope(
            contamination=CONTAMINATION,
            random_state=42
        )
        model.fit(X)
        
        logger.info("Elliptic Envelope training complete")
        return model
    
    def train_all_models(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Train all anomaly detection models
        """
        logger.info(f"Training all models on {X.shape[0]} samples with {X.shape[1]} features")
        
        self.models['isolation_forest'] = self.train_isolation_forest(X)
        self.models['local_outlier_factor'] = self.train_local_outlier_factor(X)
        self.models['one_class_svm'] = self.train_one_class_svm(X)
        self.models['elliptic_envelope'] = self.train_elliptic_envelope(X)
        
        self.trained = True
        logger.info(f"All {len(self.models)} models trained successfully")
        
        return self.models
    
    def predict(self, X: np.ndarray, model_name: str = 'isolation_forest') -> np.ndarray:
        """
        Predict anomalies using specified model
        
        Returns:
            Array of predictions: 1 for normal, -1 for anomaly
        """
        if not self.trained:
            raise ValueError("Models not trained. Call train_all_models() first.")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if model_name == 'local_outlier_factor':
            predictions = model.predict(X)
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def predict_ensemble(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        Ensemble prediction: combine multiple models
        
        Args:
            X: Feature matrix
            threshold: Proportion of models that must agree (0.5 = majority vote)
        
        Returns:
            (predictions, model_votes)
        """
        if not self.trained:
            raise ValueError("Models not trained. Call train_all_models() first.")
        
        logger.info(f"Running ensemble prediction with {len(self.models)} models")
        
        # Get predictions from all models
        all_predictions = {}
        for model_name in self.models.keys():
            preds = self.predict(X, model_name)
            # Convert to binary: 1 = anomaly, 0 = normal
            all_predictions[model_name] = (preds == -1).astype(int)
        
        # Combine predictions
        pred_matrix = np.array(list(all_predictions.values()))
        anomaly_votes = pred_matrix.sum(axis=0)
        
        # Threshold for ensemble decision
        num_models = len(self.models)
        ensemble_predictions = (anomaly_votes >= (num_models * threshold)).astype(int)
        
        # Convert back to -1/1 format
        ensemble_predictions = np.where(ensemble_predictions == 1, -1, 1)
        
        model_votes = {
            'votes_per_sample': anomaly_votes.tolist(),
            'individual_predictions': all_predictions,
            'ensemble_threshold': threshold,
            'total_models': num_models
        }
        
        logger.info(f"Ensemble detected {(ensemble_predictions == -1).sum()} anomalies")
        
        return ensemble_predictions, model_votes
    
    def get_anomaly_scores(self, X: np.ndarray, model_name: str = 'isolation_forest') -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'score_samples'):
            scores = model.score_samples(X)
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
        else:
            scores = None
        
        return scores
    
    def analyze_anomalies(self, X: np.ndarray, df: pd.DataFrame, 
                          model_name: str = 'isolation_forest') -> pd.DataFrame:
        """
        Analyze detected anomalies with context
        
        Returns:
            DataFrame with anomaly information
        """
        predictions = self.predict(X, model_name)
        scores = self.get_anomaly_scores(X, model_name)
        
        # Create results dataframe
        results = df.copy()
        results['is_anomaly'] = (predictions == -1).astype(int)
        results['anomaly_score'] = scores if scores is not None else np.nan
        
        # Filter to anomalies only
        anomalies = results[results['is_anomaly'] == 1].copy()
        
        # Sort by score (most anomalous first)
        if scores is not None:
            anomalies = anomalies.sort_values('anomaly_score')
        
        logger.info(f"Found {len(anomalies)} anomalies using {model_name}")
        
        return anomalies
    
    def get_anomaly_summary(self, anomalies: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for anomalies
        """
        if len(anomalies) == 0:
            return {"message": "No anomalies detected"}
        
        summary = {
            'total_anomalies': len(anomalies),
            'percentage': round(len(anomalies) / len(anomalies) * 100, 2),
            'by_state': anomalies['ST'].value_counts().head(5).to_dict(),
            'by_coverage_type': anomalies['TYP'].value_counts().to_dict(),
            'by_plan_type': anomalies['TP1'].value_counts().to_dict(),
            'by_status': anomalies['STS'].value_counts().to_dict(),
            'avg_age': round(anomalies['AGE'].mean(), 1),
            'avg_family_size': round(anomalies['FAMILY_SIZE'].mean(), 2),
            'cancellation_rate': round((anomalies['IS_CANCELLED'].sum() / len(anomalies)) * 100, 2)
        }
        
        return summary
    
    def save_models(self, save_dir: Path = MODELS_DIR):
        """Save trained models to disk"""
        if not self.trained:
            raise ValueError("No models to save. Train models first.")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = save_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        logger.info(f"All models saved to {save_dir}")
    
    def load_models(self, load_dir: Path = MODELS_DIR):
        """Load trained models from disk"""
        load_dir = Path(load_dir)
        
        model_files = list(load_dir.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {load_dir}")
        
        for model_file in model_files:
            model_name = model_file.stem
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} from {model_file}")
        
        self.trained = True
        logger.info(f"Loaded {len(self.models)} models from {load_dir}")


class AnomalyExplainer:
    """
    Explain why samples are flagged as anomalies
    """
    
    @staticmethod
    def explain_anomaly(row: pd.Series) -> List[str]:
        """
        Generate human-readable explanations for anomalies based on domain-specific business rules
        """
        reasons = []
        
        # Critical data quality issues
        if row.get('MISSING_CERT', 0) == 1:
            reasons.append("⚠️ Missing CERT (Certificate Number) - Critical data quality issue")
        if row.get('MISSING_SSN', 0) == 1:
            reasons.append("⚠️ Missing SSN - Critical identifier missing")
        if row.get('MISSING_HCID', 0) == 1:
            reasons.append("⚠️ Missing HCID (Health Care ID) - Critical identifier missing")
        
        # Member code (MCDE) anomalies
        if row.get('PRIMARY_CHILD', 0) == 1:
            reasons.append(f"🔴 Primary member (MCDE={row.get('MCDE')}) is under 18 years old (Age: {row.get('AGE')})")
        
        mcde = row.get('MCDE', 0)
        if mcde not in [10, 20, 30, 40, 50, 60, 70] and mcde > 0:
            reasons.append(f"❌ Invalid MCDE code: {mcde} (Expected: 10, 20, 30-70)")
        
        # Business type (MBUTY) inconsistencies
        if row.get('IND_MULTI_MEMBER', 0) == 1:
            reasons.append(f"🔴 Individual (IND) plan with multiple members (MCNT={row.get('FAMILY_SIZE')})")
        
        if row.get('SMGRP_SINGLE_MEMBER', 0) == 1:
            reasons.append("🔴 Small Group (SMGRP) plan with single member (MCNT=1)")
        
        # Cancellation reason (CR) anomalies
        cr = row.get('CR')
        if cr == 11:  # Never Effective
            if row.get('NEVER_EFF_LONG_DURATION', 0) == 1:
                reasons.append(f"⚠️ Never Effective (CR=11) but contract duration > 30 days ({row.get('CONTRACT_DURATION')} days)")
            else:
                reasons.append("📋 Never Effective cancellation (CR=11)")
        elif cr == 8:  # Non-Payment
            reasons.append("💰 Non-Payment cancellation (CR=8)")
        elif cr == 67:  # Death
            reasons.append("⚫ Death cancellation (CR=67)")
        
        # Date anomalies
        if row.get('IMMEDIATE_CANCEL', 0) == 1:
            reasons.append("⚡ Immediate cancellation (CONT EFF = CONT CAN on same day)")
        
        if row.get('CANCEL_DATE_MISMATCH', 0) == 1:
            reasons.append("📅 Contract cancel date ≠ Member cancel date (CONT CAN ≠ MEMCANDT)")
        
        if row.get('FUTURE_EFFECTIVE', 0) == 1:
            reasons.append("🔮 Future effective date (CONT EFF is in the future)")
        
        if row.get('CONTRACT_DURATION', 0) < 0:
            reasons.append("❌ Negative contract duration (Invalid date sequence)")
        
        # Early cancellation
        if row.get('DAYS_TO_CANCEL', 0) > 0 and row.get('DAYS_TO_CANCEL', 0) < 30:
            reasons.append(f"⏱️ Cancelled within 30 days ({int(row.get('DAYS_TO_CANCEL'))} days)")
        
        # Age-related
        age = row.get('AGE', 0)
        if age > 100:
            reasons.append(f"👴 Age exceeds 100 years (Age: {age})")
        elif age < 0:
            reasons.append(f"❌ Negative age (Age: {age}) - Data quality issue")
        
        # Family size
        family_size = row.get('FAMILY_SIZE', 0)
        if family_size > 15:
            reasons.append(f"👨‍👩‍👧‍👦 Unusually large family (MCNT={int(family_size)})")
        elif family_size < 1:
            reasons.append(f"❌ Invalid family size (MCNT={family_size})")
        
        # Plan type mismatches
        typ = row.get('TYP')
        tp1 = row.get('TP1')
        if typ == 'MED' and tp1 not in ['HMO', 'PPO', 'POS', 'EPO', None]:
            reasons.append(f"⚠️ Medical coverage with unusual plan type: {tp1}")
        
        # Exchange indicator (XI) patterns
        xi = row.get('XI', '')
        mbuty = row.get('MBUTY', '')
        if xi in ['PB', 'PR', 'PO'] and mbuty == 'SMGRP':
            reasons.append(f"⚠️ Exchange indicator ({xi}) with Small Group - unusual pattern")
        
        # Status (STS) inconsistencies
        sts = row.get('STS')
        if sts == 0 and row.get('IS_CANCELLED', 0) == 1:
            reasons.append("⚠️ Status=Active (0) but member has cancellation data")
        
        # Statistical outlier (if no specific reasons found)
        if not reasons:
            reasons.append("📊 Statistical outlier based on multiple features")
        
        return reasons
    
    @staticmethod
    def add_explanations(anomalies: pd.DataFrame) -> pd.DataFrame:
        """
        Add explanation column to anomalies dataframe
        """
        anomalies = anomalies.copy()
        anomalies['anomaly_reasons'] = anomalies.apply(
            AnomalyExplainer.explain_anomaly, axis=1
        )
        return anomalies
