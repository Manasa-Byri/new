"""
Data Preprocessing Pipeline for Anomaly Detection
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class InsuranceDataPreprocessor:
    """
    Preprocessor for insurance member data
    Handles missing values, feature engineering, and encoding
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for anomaly detection"""
        df = df.copy()
        
        # Date conversions
        df['MEMBIRDT_DATE'] = pd.to_datetime(df['MEMBIRDT'].astype(str), format='%Y%m%d', errors='coerce')
        df['CONT_EFF_DATE'] = pd.to_datetime(df['CONT EFF'].astype(str), format='%Y%m%d', errors='coerce')
        df['MEMEFFDT_DATE'] = pd.to_datetime(df['MEMEFFDT'].astype(str), format='%Y%m%d', errors='coerce')
        
        # Handle cancellation dates (0 means not cancelled)
        df['CONT_CAN_DATE'] = df['CONT CAN'].apply(
            lambda x: pd.to_datetime(str(x), format='%Y%m%d', errors='coerce') if x != 0 else None
        )
        df['MEMCANDT_DATE'] = df['MEMCANDT'].apply(
            lambda x: pd.to_datetime(str(x), format='%Y%m%d', errors='coerce') if x != 0 else None
        )
        
        # Age calculation
        df['AGE'] = ((datetime.now() - df['MEMBIRDT_DATE']).dt.days / 365.25).fillna(0).astype(int)
        
        # Family size (MCNT)
        df['FAMILY_SIZE'] = df['MCNT'].fillna(1)
        
        # Contract duration (days)
        df['CONTRACT_DURATION'] = (df['MEMCANDT_DATE'] - df['MEMEFFDT_DATE']).dt.days
        df['CONTRACT_DURATION'] = df['CONTRACT_DURATION'].fillna(
            (datetime.now() - df['MEMEFFDT_DATE']).dt.days
        )
        
        # Time to cancellation (for cancelled members)
        df['DAYS_TO_CANCEL'] = (df['CONT_CAN_DATE'] - df['CONT_EFF_DATE']).dt.days
        df['DAYS_TO_CANCEL'] = df['DAYS_TO_CANCEL'].fillna(0)
        
        # Binary flags
        df['IS_CANCELLED'] = (df['STS'] == 1).astype(int)
        df['IS_MEDICAL'] = (df['TYP'] == 'MED').astype(int)
        df['IS_DENTAL'] = (df['TYP'] == 'DEN').astype(int)
        df['IS_PPO'] = (df['TP1'] == 'PPO').astype(int)
        df['IS_HMO'] = (df['TP1'] == 'HMO').astype(int)
        df['IS_SMALL_GROUP'] = (df['MBUTY'] == 'SMGRP').astype(int)
        df['IS_INDIVIDUAL'] = (df['MBUTY'] == 'IND').astype(int)
        
        # Age groups
        df['AGE_GROUP'] = pd.cut(df['AGE'], 
                                  bins=[0, 18, 26, 35, 45, 55, 65, 100],
                                  labels=['0-17', '18-25', '26-34', '35-44', '45-54', '55-64', '65+'])
        
        # Member type (domain-specific MCDE mapping)
        from ml.config import MEMBER_CODE_MAPPING
        df['MEMBER_TYPE'] = df['MCDE'].apply(lambda x: 
            MEMBER_CODE_MAPPING.get(x, f'Dependent_{x//10}' if x >= 30 else 'Unknown')
        )
        
        # Is primary member (MCDE 10 or 20)
        df['IS_PRIMARY'] = df['MCDE'].isin([10, 20]).astype(int)
        
        # Cancellation reason category (domain-specific CR mapping)
        from ml.config import CANCEL_REASON_MAPPING
        df['CANCEL_REASON_CAT'] = df['CR'].apply(lambda x:
            CANCEL_REASON_MAPPING.get(x, 'Other') if pd.notna(x) else 'None'
        )
        
        # Exchange indicator category
        from ml.config import EXCHANGE_INDICATOR_MAPPING
        df['EXCHANGE_CAT'] = df['XI'].fillna('').apply(lambda x:
            EXCHANGE_INDICATOR_MAPPING.get(x, 'Unknown')
        )
        
        # Business code category
        from ml.config import BUSINESS_CODE_MAPPING
        df['BUSINESS_CODE_CAT'] = df['BUS'].fillna('').apply(lambda x:
            BUSINESS_CODE_MAPPING.get(x, 'Unknown')
        )
        
        # Data quality flags
        df['MISSING_CERT'] = df['CERT'].isna().astype(int)
        df['MISSING_SSN'] = df['SSN'].isna().astype(int)
        df['MISSING_HCID'] = df['HCID'].isna().astype(int)
        
        # Business rule violations
        # 1. Individual (IND) should typically have MCNT=1
        df['IND_MULTI_MEMBER'] = ((df['MBUTY'] == 'IND') & (df['MCNT'] > 1)).astype(int)
        
        # 2. Small group should have multiple members
        df['SMGRP_SINGLE_MEMBER'] = ((df['MBUTY'] == 'SMGRP') & (df['MCNT'] == 1)).astype(int)
        
        # 3. Primary member (10/20) should be adult for most cases
        df['PRIMARY_CHILD'] = ((df['MCDE'].isin([10, 20])) & (df['AGE'] < 18)).astype(int)
        
        # 4. Contract cancel and member cancel date mismatch
        df['CANCEL_DATE_MISMATCH'] = (
            (df['CONT_CAN_DATE'].notna()) & 
            (df['MEMCANDT_DATE'].notna()) & 
            (df['CONT_CAN_DATE'] != df['MEMCANDT_DATE'])
        ).astype(int)
        
        # 5. Immediate cancellation (same day as effective)
        df['IMMEDIATE_CANCEL'] = (
            (df['CONT_EFF_DATE'] == df['CONT_CAN_DATE']) & 
            (df['CONT_CAN_DATE'].notna())
        ).astype(int)
        
        # 6. Future effective date
        df['FUTURE_EFFECTIVE'] = (df['CONT_EFF_DATE'] > datetime.now()).astype(int)
        
        # 7. Never effective with long duration
        df['NEVER_EFF_LONG_DURATION'] = (
            (df['CR'] == 11) & 
            (df['CONTRACT_DURATION'] > 30)
        ).astype(int)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Prepare features for ML models
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders/scalers (True for training, False for inference)
        
        Returns:
            Scaled feature matrix
        """
        df = df.copy()
        
        # Select numeric features (including domain-specific flags)
        numeric_cols = [
            'AGE', 'FAMILY_SIZE', 'CONTRACT_DURATION', 'DAYS_TO_CANCEL',
            'IS_CANCELLED', 'IS_MEDICAL', 'IS_DENTAL', 'IS_PPO', 'IS_HMO',
            'IS_SMALL_GROUP', 'IS_INDIVIDUAL', 'IS_PRIMARY',
            'MISSING_CERT', 'MISSING_SSN', 'MISSING_HCID',
            'IND_MULTI_MEMBER', 'SMGRP_SINGLE_MEMBER', 'PRIMARY_CHILD',
            'CANCEL_DATE_MISMATCH', 'IMMEDIATE_CANCEL', 'FUTURE_EFFECTIVE',
            'NEVER_EFF_LONG_DURATION'
        ]
        
        # Select categorical features to encode
        categorical_cols = [
            'ST', 'TYP', 'TP1', 'MBUTY', 'MEMBER_TYPE', 
            'CANCEL_REASON_CAT', 'EXCHANGE_CAT', 'BUSINESS_CODE_CAT'
        ]
        
        # Handle missing values in numeric columns
        numeric_data = df[numeric_cols].fillna(0)
        
        # Encode categorical features
        encoded_features = []
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[col].fillna('UNKNOWN').astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        # Handle unseen categories
                        df[col] = df[col].fillna('UNKNOWN').astype(str)
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
                        encoded = le.transform(df[col])
                    else:
                        encoded = np.zeros(len(df))
                
                encoded_features.append(encoded.reshape(-1, 1))
        
        # Combine numeric and encoded categorical features
        if encoded_features:
            categorical_data = np.hstack(encoded_features)
            feature_matrix = np.hstack([numeric_data.values, categorical_data])
        else:
            feature_matrix = numeric_data.values
        
        # Scale features
        if fit:
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            self.feature_names = numeric_cols + categorical_cols
        else:
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        logger.info(f"Prepared feature matrix: {feature_matrix_scaled.shape}")
        return feature_matrix_scaled
    
    def preprocess(self, file_path: str, fit: bool = True) -> tuple:
        """
        Complete preprocessing pipeline
        
        Returns:
            (feature_matrix, original_dataframe)
        """
        # Load data
        df = self.load_data(file_path)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare features for ML
        X = self.prepare_features(df, fit=fit)
        
        return X, df
    
    def get_feature_importance_names(self) -> list:
        """Get feature names for interpretation"""
        return self.feature_names
