# Machine Learning Anomaly Detection for Insurance Data

## Overview

This ML module implements **anomaly detection** for the insurance member dataset using multiple machine learning algorithms. It identifies unusual patterns, outliers, and potentially fraudulent or erroneous records.

---

## 📁 Folder Structure

```
ml/
├── __init__.py                 # Module initialization
├── config.py                   # Configuration and parameters
├── data_preprocessing.py       # Data preprocessing pipeline
├── anomaly_detector.py         # ML models for anomaly detection
├── train.py                    # Training script
├── inference.py                # Inference service
├── README.md                   # This file
├── models/                     # Trained models (generated)
│   ├── isolation_forest.joblib
│   ├── local_outlier_factor.joblib
│   ├── one_class_svm.joblib
│   ├── elliptic_envelope.joblib
│   └── preprocessor.joblib
├── data/                       # Processed data (generated)
├── notebooks/                  # Jupyter notebooks for exploration
└── results/                    # Training results and reports
    ├── anomalies_YYYYMMDD_HHMMSS.csv
    └── evaluation_report_YYYYMMDD_HHMMSS.txt
```

---

## 🎯 Anomaly Types Detected

### 1. **Age-Related Anomalies**
- Members under 18 as primary subscribers
- Members over 100 years old
- Age inconsistent with member type

### 2. **Family Size Anomalies**
- Unusually large families (>10 members)
- Small group plans with single member

### 3. **Cancellation Pattern Anomalies**
- Cancellations within 30 days of enrollment
- Unusual cancellation reasons
- High cancellation rates in specific segments

### 4. **Plan Mismatch Anomalies**
- Medical coverage with unusual plan types
- Inconsistent plan and coverage combinations

### 5. **Geographic Anomalies**
- Unusual state distributions
- Outlier patterns by region

### 6. **Temporal Anomalies**
- Negative contract durations
- Invalid date sequences
- Unusual enrollment patterns

---

## 🤖 ML Models Implemented

### 1. **Isolation Forest** (Primary Model)
- **Algorithm**: Tree-based ensemble
- **Best For**: High-dimensional data, fast training
- **Contamination**: 5% (configurable)
- **Use Case**: General-purpose anomaly detection

### 2. **Local Outlier Factor (LOF)**
- **Algorithm**: Density-based
- **Best For**: Local density deviations
- **Use Case**: Identifying outliers in dense regions

### 3. **One-Class SVM**
- **Algorithm**: Support Vector Machine
- **Best For**: Non-linear decision boundaries
- **Use Case**: Complex pattern recognition

### 4. **Elliptic Envelope**
- **Algorithm**: Covariance-based
- **Best For**: Gaussian distributed data
- **Use Case**: Statistical outlier detection

### 5. **Ensemble Model** (Recommended)
- **Combines**: All 4 models
- **Method**: Majority voting
- **Threshold**: 50% agreement (configurable)
- **Use Case**: Production deployment

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `scikit-learn==1.4.0`
- `pandas==2.2.0`
- `numpy==1.26.3`
- `joblib==1.3.2`

### Step 2: Train Models

```bash
cd ml
python train.py
```

**Output:**
- Trained models saved to `ml/models/`
- Anomaly results saved to `ml/results/`
- Evaluation report generated

**Training Time:** ~2-5 minutes for 50K records

### Step 3: Use API Endpoints

Start the FastAPI server:
```bash
python run.py
```

Access endpoints at: `http://localhost:8000/docs`

---

## 📊 API Endpoints

### 1. **Detect Anomalies (Single Model)**
```
GET /api/v1/ml/anomaly-detection/detect?model=isolation_forest
```

**Parameters:**
- `model`: Model name (isolation_forest, local_outlier_factor, one_class_svm, elliptic_envelope)

**Response:**
```json
{
  "success": true,
  "model_used": "isolation_forest",
  "total_records": 49999,
  "anomalies_detected": 2500,
  "anomaly_rate": 5.0,
  "summary": {
    "total_anomalies": 2500,
    "by_state": {"CO": 450, "NY": 380},
    "by_coverage_type": {"MED": 2100, "DEN": 400},
    "avg_age": 42.5,
    "cancellation_rate": 65.2
  },
  "top_anomalies": [...]
}
```

### 2. **Detect Anomalies (Ensemble)**
```
GET /api/v1/ml/anomaly-detection/detect-ensemble?threshold=0.5
```

**Parameters:**
- `threshold`: Ensemble agreement threshold (0.0-1.0)
  - 0.5 = majority vote (2+ models agree)
  - 0.75 = strong consensus (3+ models agree)
  - 1.0 = unanimous (all 4 models agree)

**Response:**
```json
{
  "success": true,
  "model_used": "ensemble",
  "models_in_ensemble": 4,
  "ensemble_threshold": 0.5,
  "anomalies_detected": 1850,
  "anomaly_rate": 3.7,
  "top_anomalies": [
    {
      "hcid": "172T97147",
      "age": 34,
      "state": "NY",
      "coverage_type": "MED",
      "status": "Inactive",
      "model_votes": 4,
      "reasons": [
        "Cancelled within 30 days of enrollment",
        "Unusual cancellation pattern"
      ]
    }
  ]
}
```

### 3. **Get Member Anomaly Details**
```
GET /api/v1/ml/anomaly-detection/member/{hcid}
```

**Response:**
```json
{
  "success": true,
  "hcid": "172T97147",
  "member_info": {
    "age": 34,
    "state": "NY",
    "coverage_type": "MED",
    "plan_type": "HMO",
    "status": "Inactive",
    "family_size": 1
  },
  "predictions": {
    "isolation_forest": "Anomaly",
    "local_outlier_factor": "Anomaly",
    "one_class_svm": "Normal",
    "elliptic_envelope": "Anomaly"
  },
  "anomaly_scores": {
    "isolation_forest": -0.234,
    "local_outlier_factor": -1.456
  },
  "anomaly_reasons": [
    "Cancelled within 30 days of enrollment",
    "Unusual cancellation pattern for state"
  ]
}
```

### 4. **Get Statistics**
```
GET /api/v1/ml/anomaly-detection/statistics
```

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_records": 49999,
    "models_available": ["isolation_forest", "local_outlier_factor", "one_class_svm", "elliptic_envelope"],
    "model_statistics": {
      "isolation_forest": {
        "anomalies_detected": 2500,
        "anomaly_rate": 5.0
      },
      "local_outlier_factor": {
        "anomalies_detected": 2650,
        "anomaly_rate": 5.3
      }
    },
    "ensemble_statistics": {
      "anomalies_detected": 1850,
      "anomaly_rate": 3.7
    }
  }
}
```

### 5. **List Available Models**
```
GET /api/v1/ml/anomaly-detection/models
```

---

## 🔧 Configuration

Edit `ml/config.py` to customize:

```python
# Expected proportion of anomalies
CONTAMINATION = 0.05  # 5%

# Random seed for reproducibility
RANDOM_STATE = 42

# Features to use
NUMERIC_FEATURES = ['AGE', 'FAMILY_SIZE', ...]
CATEGORICAL_FEATURES = ['ST', 'TYP', 'TP1', ...]

# Model parameters
MODELS_CONFIG = {
    'isolation_forest': {
        'n_estimators': 100,
        'contamination': 0.05,
        ...
    }
}
```

---

## 📈 Feature Engineering

### Numeric Features
- `AGE`: Calculated from birth date
- `FAMILY_SIZE`: Number of family members
- `CONTRACT_DURATION`: Days enrolled
- `DAYS_TO_CANCEL`: Time to cancellation
- Binary flags: `IS_MEDICAL`, `IS_PPO`, `IS_SMALL_GROUP`, etc.

### Categorical Features (Encoded)
- `ST`: State
- `TYP`: Coverage type
- `TP1`: Plan type
- `MBUTY`: Business type
- `MEMBER_TYPE`: Primary/Dependent
- `CANCEL_REASON_CAT`: Cancellation category

### Preprocessing Steps
1. Date parsing and conversion
2. Age calculation
3. Feature engineering (derived fields)
4. Missing value imputation
5. Label encoding for categorical features
6. Standard scaling (zero mean, unit variance)

---

## 🎓 Model Training Details

### Training Process
1. **Load Data**: Read CSV file
2. **Preprocess**: Engineer features, handle missing values
3. **Train Models**: Fit all 4 anomaly detection models
4. **Evaluate**: Generate predictions and metrics
5. **Ensemble**: Combine model predictions
6. **Explain**: Add human-readable explanations
7. **Save**: Store models and results

### Evaluation Metrics
- **Anomaly Count**: Total anomalies detected
- **Anomaly Rate**: Percentage of dataset
- **State Distribution**: Anomalies by state
- **Coverage Distribution**: Anomalies by coverage type
- **Cancellation Rate**: % of anomalies that are cancelled

### Output Files
- `models/*.joblib`: Trained model files
- `results/anomalies_*.csv`: Detected anomalies with details
- `results/evaluation_report_*.txt`: Training summary

---

## 💡 Usage Examples

### Python Script
```python
from ml.inference import get_anomaly_service

# Initialize service
service = get_anomaly_service()

# Detect anomalies with ensemble
result = service.detect_anomalies_ensemble(threshold=0.5)

print(f"Found {result['anomalies_detected']} anomalies")
print(f"Anomaly rate: {result['anomaly_rate']}%")

# Get details for specific member
details = service.get_anomaly_details('172T97147')
print(f"Member status: {details['member_info']['status']}")
print(f"Anomaly reasons: {details['anomaly_reasons']}")
```

### cURL
```bash
# Detect anomalies
curl http://localhost:8000/api/v1/ml/anomaly-detection/detect-ensemble

# Get member details
curl http://localhost:8000/api/v1/ml/anomaly-detection/member/172T97147

# Get statistics
curl http://localhost:8000/api/v1/ml/anomaly-detection/statistics
```

---

## 🔍 Interpreting Results

### Anomaly Score
- **Lower scores** = More anomalous
- **Negative scores** = Likely anomaly
- **Positive scores** = Likely normal

### Model Votes (Ensemble)
- **4 votes**: All models agree (high confidence)
- **3 votes**: Strong consensus
- **2 votes**: Majority agreement
- **1 vote**: Weak signal
- **0 votes**: Normal

### Anomaly Reasons
Human-readable explanations for why a record is flagged:
- "Cancelled within 30 days of enrollment"
- "Unusually large family size: 15"
- "Age exceeds 100 years"
- "Statistical outlier based on multiple features"

---

## 🎯 Best Practices

### For Production Use
1. **Use Ensemble Model** with threshold=0.5
2. **Review High-Vote Anomalies** first (3-4 votes)
3. **Investigate Patterns** not individual records
4. **Retrain Periodically** as data changes
5. **Monitor False Positives** and adjust contamination

### For Investigation
1. **Start with Statistics** endpoint
2. **Filter by State/Type** to find patterns
3. **Check Individual Members** for details
4. **Review Anomaly Reasons** for insights
5. **Cross-reference** with business rules

### For Tuning
1. **Adjust Contamination** if too many/few anomalies
2. **Modify Threshold** for ensemble sensitivity
3. **Add/Remove Features** based on domain knowledge
4. **Experiment with Models** for specific use cases

---

## 🐛 Troubleshooting

### Models Not Found
```
Error: No model files found in ml/models/
```
**Solution**: Run `python ml/train.py` to train models first

### Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce dataset size or increase system memory

### Poor Performance
- **Too many anomalies**: Increase `CONTAMINATION` in config
- **Too few anomalies**: Decrease `CONTAMINATION` in config
- **False positives**: Use ensemble with higher threshold (0.75)

---

## 📚 References

### Algorithms
- **Isolation Forest**: Liu et al. (2008) - "Isolation Forest"
- **LOF**: Breunig et al. (2000) - "LOF: Identifying Density-Based Local Outliers"
- **One-Class SVM**: Schölkopf et al. (2001) - "Estimating the Support of a High-Dimensional Distribution"

### Libraries
- **scikit-learn**: https://scikit-learn.org/stable/modules/outlier_detection.html
- **pandas**: https://pandas.pydata.org/
- **numpy**: https://numpy.org/

---

## 🚀 Future Enhancements

### Phase 1 (Current)
- ✅ Multiple anomaly detection models
- ✅ Ensemble predictions
- ✅ API endpoints
- ✅ Anomaly explanations

### Phase 2 (Planned)
- [ ] Deep learning models (Autoencoders)
- [ ] Time-series anomaly detection
- [ ] Real-time streaming detection
- [ ] Anomaly clustering and categorization

### Phase 3 (Future)
- [ ] Supervised learning with labeled anomalies
- [ ] Fraud detection specific models
- [ ] Automated retraining pipeline
- [ ] A/B testing framework

---

## 📞 Support

For questions or issues:
1. Check this README
2. Review training logs in `ml/results/`
3. Check API documentation at `/docs`
4. Review code comments in source files

---

## 📄 License

Part of the Insight Agent Work project.

---

**Last Updated**: June 4, 2026  
**Version**: 1.0  
**Status**: Production Ready
