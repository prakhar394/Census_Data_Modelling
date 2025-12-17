# JPMC Census Income Classification & Customer Segmentation

## Project Overview

This project addresses two critical business objectives for retail marketing:

1. Income Classification: Predict whether a person earns above or below $50,000 annually using demographic and employment variables
2. Customer Segmentation: Create actionable customer segments for targeted marketing strategies

Dataset: U.S. Census Bureau Current Population Survey (1994-1995)
- 199,523 observations
- 42 demographic and employment variables
- Stratified sampling weights included

## Project Structure

```
jpmc_census_project/
├── data/
│   ├── census-bureau.data        # Raw data file (place here)
│   └── census-bureau.columns      # Column definitions (place here)
├── models/
│   └── [trained models saved here]
├── outputs/
│   └── [results and predictions saved here]
├── scripts/
│   ├── config.py                  # Configuration and hyperparameters
│   ├── data_preprocessing.py      # Data loading and cleaning
│   ├── utils.py                   # Evaluation utilities
│   ├── classification_model.py    # Deliverable 1: Classification training
│   ├── segmentation_model.py      # Deliverable 2: Segmentation model
│   └── main.py                    # Run complete pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── JPMC_Project_Report.docx      # Deliverable 4: Project report
```

## Quick Start

### 1. System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: ~500MB for data and models
- **OS**: Linux, macOS, or Windows

### 2. Installation

```bash
# Clone or extract the project
cd jpmc_census_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Setup

**IMPORTANT**: Place your data files in the `data/` directory:
- `census-bureau.data` - Main data file
- `census-bureau.columns` - Column names file

**Verify data location**:
```bash
ls data/
# Should show: census-bureau.columns  census-bureau.data
```

### 4. Run Complete Pipeline

```bash
cd scripts
python main.py
```

This will:
1. Load and preprocess data (~5 min)
2. Train 4 classification models (~20-30 min)
3. Generate customer segmentation (~10 min)
4. Save all models and results

**Expected Output**:
- Trained models in `models/`
- Performance metrics printed to console
- Segmentation profiles saved to `outputs/`

## Running Individual Components

### Classification Only (Deliverable 1)

Train and evaluate income classification models:

```bash
cd scripts
python classification_model.py
```

**What it does**:
- Loads and preprocesses data
- Trains 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluates with ROC-AUC, PR-AUC, Precision, Recall, F1
- Finds optimal decision threshold
- Saves trained models to `models/`

**Expected Runtime**: 20-30 minutes

**Output Files**:
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/xgboost.pkl`
- `models/lightgbm.pkl`

### Segmentation Only (Deliverable 2)

Generate customer segments:

```bash
cd scripts
python segmentation_model.py
```

**What it does**:
- Filters data to adults (age ≥ 18)
- Applies dimensionality reduction (TruncatedSVD)
- Finds optimal number of clusters via silhouette score
- Creates weighted K-Means segments
- Profiles each segment with key characteristics
- Saves segmentation model and results

**Expected Runtime**: 10-15 minutes

**Output Files**:
- `models/segmentation_model.pkl`
- `outputs/segmented_customers.csv`
- `outputs/segment_profiles.csv`

### Data Preprocessing Only

Test data loading and preprocessing:

```bash
cd scripts
python data_preprocessing.py
```

## Configuration

Edit `scripts/config.py` to customize:

### File Paths
```python
DATA_PATH = "../data/census-bureau.data"
COLS_PATH = "../data/census-bureau.columns"
```

### Model Hyperparameters
```python
XGBOOST_PARAMS = {
    'max_depth': 8,
    'learning_rate': 0.03,
    'n_estimators': 700,
    ...
}
```

### Segmentation Parameters
```python
SEGMENTATION_PARAMS = {
    'svd_components': 40,
    'k_min': 3,
    'k_max': 7,
    ...
}
```

### Getting Help

If you encounter issues:
1. Check error message carefully
2. Verify all dependencies installed: `pip list`
3. Ensure data files are in correct location
4. Review console output for warnings

## Code Files Description

### Deliverable 1: Classification (classification_model.py)
- Purpose: Train and evaluate income prediction models
- Key Functions:
  - train_logistic_regression(): Trains LR baseline
  - train_xgboost(): Trains XGBoost model (recommended)
  - evaluate_all_models(): Compares model performance
  - main(): Complete classification pipeline
- Outputs: Trained models, performance metrics

### Deliverable 2: Segmentation (segmentation_model.py)
- Purpose: Create customer segments for marketing
- Key Functions:
  - prepare_segmentation_data(): Filters to adults
  - find_optimal_k(): Determines best number of segments
  - create_segments(): Runs weighted K-Means
  - profile_segments(): Generates segment descriptions
  - main(): Complete segmentation pipeline
- Outputs: Segment model, profiles, assignments

### Supporting Files
- data_preprocessing.py: Data loading, cleaning, feature engineering
- utils.py: Evaluation metrics, threshold optimization
- config.py: Centralized configuration
- main.py: Orchestrates complete pipeline

## Project Report

For detailed analysis, methodology, and business insights, see:
JPMC_Project_Report.docx

The report includes:
- Data exploration and preprocessing
- Model architecture and training approach
- Evaluation procedures and results
- Segment profiles and marketing strategies
- Business recommendations
- References


## Deliverables Checklist

- Deliverable 1: Classification model code (classification_model.py)
- Deliverable 2: Segmentation model code (segmentation_model.py)
- Deliverable 3: README with execution instructions (this file)
- Deliverable 4: Project report (JPMC_Project_Report.docx)

## References

1. Scikit-learn Documentation - https://scikit-learn.org
2. XGBoost Documentation - https://xgboost.readthedocs.io
3. LightGBM Documentation - https://lightgbm.readthedocs.io
4. U.S. Census Bureau - https://www.census.gov
5. Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"

