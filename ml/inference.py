"""
Inference Script for Anomaly Detection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, List

from ml.config import CSV_FILE, MODELS_DIR
from ml.anomaly_detector import AnomalyDetector, AnomalyExplainer

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """
    Service for real-time anomaly detection
    """
    
    def __init__(self):
        self.detector = None
        self.preprocessor = None
        self.loaded = False
        
    def load_models(self):
        """Load trained models and preprocessor"""
        try:
            logger.info("Loading anomaly detection models...")
            
            # Load detector
            self.detector = AnomalyDetector()
            self.detector.load_models(MODELS_DIR)
            
            # Load preprocessor
            preprocessor_path = MODELS_DIR / "preprocessor.joblib"
            self.preprocessor = joblib.load(preprocessor_path)
            
            self.loaded = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def detect_anomalies(self, model_name: str = 'isolation_forest') -> Dict[str, Any]:
        """
        Detect anomalies in the dataset
        
        Args:
            model_name: Model to use for detection
        
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.loaded:
            self.load_models()
        
        try:
            # Preprocess data
            X, df = self.preprocessor.preprocess(str(CSV_FILE), fit=False)
            
            # Detect anomalies
            anomalies_df = self.detector.analyze_anomalies(X, df, model_name)
            
            # Add explanations
            anomalies_df = AnomalyExplainer.add_explanations(anomalies_df)
            
            # Get summary
            summary = self.detector.get_anomaly_summary(anomalies_df)
            
            # Prepare response
            result = {
                'success': True,
                'model_used': model_name,
                'total_records': len(df),
                'anomalies_detected': len(anomalies_df),
                'anomaly_rate': round((len(anomalies_df) / len(df)) * 100, 2),
                'summary': summary,
                'top_anomalies': self._format_top_anomalies(anomalies_df, limit=20)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def detect_anomalies_ensemble(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect anomalies using ensemble of models
        """
        if not self.loaded:
            self.load_models()
        
        try:
            # Preprocess data
            X, df = self.preprocessor.preprocess(str(CSV_FILE), fit=False)
            
            # Ensemble prediction
            predictions, model_votes = self.detector.predict_ensemble(X, threshold)
            
            # Get anomalies
            anomalies_df = df[predictions == -1].copy()
            anomalies_df['votes'] = [model_votes['votes_per_sample'][i] 
                                      for i in anomalies_df.index]
            
            # Add explanations
            anomalies_df = AnomalyExplainer.add_explanations(anomalies_df)
            
            # Sort by votes (most agreed upon anomalies first)
            anomalies_df = anomalies_df.sort_values('votes', ascending=False)
            
            # Get summary
            summary = self.detector.get_anomaly_summary(anomalies_df)
            
            result = {
                'success': True,
                'model_used': 'ensemble',
                'models_in_ensemble': model_votes['total_models'],
                'ensemble_threshold': threshold,
                'total_records': len(df),
                'anomalies_detected': len(anomalies_df),
                'anomaly_rate': round((len(anomalies_df) / len(df)) * 100, 2),
                'summary': summary,
                'top_anomalies': self._format_top_anomalies(anomalies_df, limit=20)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble anomaly detection failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_anomaly_details(self, hcid: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific member
        """
        if not self.loaded:
            self.load_models()
        
        try:
            # Preprocess data
            X, df = self.preprocessor.preprocess(str(CSV_FILE), fit=False)
            
            # Find member
            member = df[df['HCID'] == hcid]
            if len(member) == 0:
                return {
                    'success': False,
                    'error': f'Member {hcid} not found'
                }
            
            # Get member index
            idx = member.index[0]
            member_features = X[idx:idx+1]
            
            # Check if anomaly
            predictions = {}
            scores = {}
            for model_name in self.detector.models.keys():
                pred = self.detector.predict(member_features, model_name)[0]
                predictions[model_name] = 'Anomaly' if pred == -1 else 'Normal'
                
                score = self.detector.get_anomaly_scores(member_features, model_name)
                if score is not None:
                    scores[model_name] = float(score[0])
            
            # Get explanations if anomaly
            member_data = member.iloc[0]
            reasons = AnomalyExplainer.explain_anomaly(member_data)
            
            result = {
                'success': True,
                'hcid': hcid,
                'member_info': {
                    'age': int(member_data.get('AGE', 0)),
                    'state': member_data.get('ST', 'N/A'),
                    'coverage_type': member_data.get('TYP', 'N/A'),
                    'plan_type': member_data.get('TP1', 'N/A'),
                    'business_type': member_data.get('MBUTY', 'N/A'),
                    'status': 'Active' if member_data.get('STS') == 0 else 'Inactive',
                    'family_size': int(member_data.get('FAMILY_SIZE', 0))
                },
                'predictions': predictions,
                'anomaly_scores': scores,
                'anomaly_reasons': reasons
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get anomaly details: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """
        Get overall anomaly detection statistics
        """
        if not self.loaded:
            self.load_models()
        
        try:
            # Preprocess data
            X, df = self.preprocessor.preprocess(str(CSV_FILE), fit=False)
            
            # Get predictions from all models
            stats = {
                'total_records': len(df),
                'models_available': list(self.detector.models.keys()),
                'model_statistics': {}
            }
            
            for model_name in self.detector.models.keys():
                predictions = self.detector.predict(X, model_name)
                num_anomalies = (predictions == -1).sum()
                
                stats['model_statistics'][model_name] = {
                    'anomalies_detected': int(num_anomalies),
                    'anomaly_rate': round((num_anomalies / len(df)) * 100, 2)
                }
            
            # Ensemble statistics
            ensemble_preds, _ = self.detector.predict_ensemble(X)
            num_ensemble_anomalies = (ensemble_preds == -1).sum()
            
            stats['ensemble_statistics'] = {
                'anomalies_detected': int(num_ensemble_anomalies),
                'anomaly_rate': round((num_ensemble_anomalies / len(df)) * 100, 2)
            }
            
            return {
                'success': True,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_top_anomalies(self, anomalies_df: pd.DataFrame, limit: int = 20) -> List[Dict]:
        """Format top anomalies for API response"""
        top_anomalies = []
        
        for idx, row in anomalies_df.head(limit).iterrows():
            anomaly = {
                'hcid': row.get('HCID', 'N/A'),
                'age': int(row.get('AGE', 0)),
                'state': row.get('ST', 'N/A'),
                'coverage_type': row.get('TYP', 'N/A'),
                'plan_type': row.get('TP1', 'N/A'),
                'status': 'Active' if row.get('STS') == 0 else 'Inactive',
                'family_size': int(row.get('FAMILY_SIZE', 0)),
                'reasons': row.get('anomaly_reasons', [])
            }
            
            if 'anomaly_score' in row and pd.notna(row['anomaly_score']):
                anomaly['anomaly_score'] = float(row['anomaly_score'])
            
            if 'votes' in row:
                anomaly['model_votes'] = int(row['votes'])
            
            top_anomalies.append(anomaly)
        
        return top_anomalies


# Global service instance
_service = None

def get_anomaly_service() -> AnomalyDetectionService:
    """Get or create anomaly detection service instance"""
    global _service
    if _service is None:
        _service = AnomalyDetectionService()
    return _service
