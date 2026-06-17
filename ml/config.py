"""
ML Configuration for Anomaly Detection
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent  # project root where CSV lives
ML_DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
ML_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data file
CSV_FILE = DATA_DIR / "USRW.NONX.IYM551ND.MEMBER.SWEEP.G2262V.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Anomaly detection thresholds
CONTAMINATION = 0.05  # Expected proportion of anomalies (5%)

# Feature engineering
NUMERIC_FEATURES = [
    'MCNT',  # Member count
    'MCDE',  # Member code
    'AGE',   # Calculated age
    'FAMILY_SIZE',  # Family size
]

CATEGORICAL_FEATURES = [
    'ST',      # State
    'TYP',     # Coverage type
    'TP1',     # Plan type 1
    'MBUTY',   # Business type
    'STS',     # Status
    'CR',      # Cancellation reason
    'ETHNI',   # Ethnicity
]

# Domain-specific business rules
MEMBER_CODE_MAPPING = {
    10: 'Primary_Male',
    20: 'Primary_Female',
    30: 'Dependent_1',
    40: 'Dependent_2',
    50: 'Dependent_3',
    60: 'Dependent_4',
    70: 'Dependent_5',
}

STATUS_MAPPING = {
    0: 'Active',
    1: 'Inactive',
    2: 'Suspended',
    4: 'Terminated'
}

CANCEL_REASON_MAPPING = {
    11: 'Never_Effective',
    8: 'Non_Payment',
    6: 'Voluntary_Termination',
    12: 'Non_Payment_Premium',
    34: 'Moved_Out_Of_Area',
    47: 'Unknown',
    67: 'Death'
}

EXCHANGE_INDICATOR_MAPPING = {
    'PB': 'Public_Exchange',
    'PR': 'Private_Exchange',
    'PO': 'Public_Other',
    'OF': 'Off_Exchange',
    '': 'Non_Exchange'
}

COVERAGE_TYPE_MAPPING = {
    'MED': 'Medical',
    'DEN': 'Dental',
    'VIS': 'Vision',
    'LFE': 'Life',
    'STD': 'Short_Term_Disability',
    'LTD': 'Long_Term_Disability'
}

PLAN_TYPE_MAPPING = {
    'HMO': 'Health_Maintenance_Organization',
    'PPO': 'Preferred_Provider_Organization',
    'POS': 'Point_Of_Service',
    'EPO': 'Exclusive_Provider_Organization'
}

BUSINESS_TYPE_MAPPING = {
    'IND': 'Individual',
    'SMGRP': 'Small_Group',
    'LGRP': 'Large_Group'
}

BUSINESS_CODE_MAPPING = {
    'B': 'Blue_Cross',
    'S': 'Special',
    'G': 'Green_State',
    'A': 'Anthem'
}

# Anomaly types to detect (domain-specific)
ANOMALY_TYPES = {
    'invalid_member_code': 'Invalid member code (MCDE) for family structure',
    'never_effective_pattern': 'Never effective cancellation (CR=11) anomalies',
    'non_payment_pattern': 'Non-payment cancellation (CR=8) patterns',
    'age_member_code_mismatch': 'Age inconsistent with member code',
    'contract_date_anomaly': 'Invalid contract date sequences',
    'cancel_date_mismatch': 'Contract cancel vs member cancel date mismatch',
    'exchange_plan_mismatch': 'Exchange indicator inconsistent with plan type',
    'family_size_anomaly': 'Unusual family size (MCNT) patterns',
    'geographic_outlier': 'Unusual state (ST) distribution',
    'business_type_inconsistency': 'MBUTY inconsistent with MCNT',
    'immediate_cancellation': 'Cancelled same day as effective',
    'future_effective_date': 'Effective date in future',
    'missing_critical_data': 'Missing CERT, HCID, or SSN',
    'duplicate_member': 'Potential duplicate member records',
}

# Model configurations
MODELS_CONFIG = {
    'isolation_forest': {
        'n_estimators': 100,
        'contamination': CONTAMINATION,
        'random_state': RANDOM_STATE,
        'max_samples': 'auto',
    },
    'local_outlier_factor': {
        'n_neighbors': 20,
        'contamination': CONTAMINATION,
        'novelty': True,
    },
    'one_class_svm': {
        'kernel': 'rbf',
        'gamma': 'auto',
        'nu': CONTAMINATION,
    },
}
