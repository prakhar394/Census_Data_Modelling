# Configuration file for JPMC Census Income Classification & Segmentation Project

# File paths
DATA_PATH = "../data/census-bureau.data"
COLS_PATH = "../data/census-bureau.columns"
OUTPUT_DIR = "../outputs"
MODELS_DIR = "../models"

# Random seed for reproducibility
RANDOM_STATE = 42

# Train-test split
TEST_SIZE = 0.2

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 0.001,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 2000,
    'class_weight': 'balanced'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced_subsample',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.03,
    'n_estimators': 700,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 5,
    'random_state': RANDOM_STATE
}

LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.06,
    'n_estimators': 500,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE
}

# Segmentation parameters
SEGMENTATION_PARAMS = {
    'svd_components': 40,
    'k_min': 3,
    'k_max': 7,
    'sample_for_silhouette': 20000
}

# Columns to drop (ultra-sparse)
DROP_SPARSE_COLS = [
    "fill_inc_questionnaire_for_veterans_admin",
    "enroll_in_edu_inst_last_wk"
]

# Log-transform columns
LOG_TRANSFORM_COLS = [
    "capital_gains",
    "dividends_from_stocks",
    "capital_losses",
    "wage_per_hour"
]

# Segmentation feature focus
SEGMENTATION_NUMERIC_FOCUS = [
    "age", "weeks_worked_in_year", "wage_per_hour",
    "capital_gains", "dividends_from_stocks"
]

SEGMENTATION_CATEGORICAL_FOCUS = [
    "education", "major_industry_code", "major_occupation_code",
    "marital_stat", "class_of_worker", "tax_filer_stat"
]
