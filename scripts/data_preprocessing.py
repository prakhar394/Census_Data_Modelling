# Data loading and preprocessing for Census Income Classification

import pandas as pd
import numpy as np
import re
from config import *


def load_column_names(cols_path=COLS_PATH):
    """Load and clean column names from columns file"""
    col_names = []
    with open(cols_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\s*\d+\s*[:\t, ]\s*(.+?)\s*$", line)
            name = m.group(1) if m else line
            name = re.sub(r"\s+", "_", name)
            name = re.sub(r"[^\w_]", "", name)
            col_names.append(name)
    
    print(f"Loaded {len(col_names)} column names")
    return col_names


def load_raw_data(data_path=DATA_PATH, col_names=None):
    """Load raw census data"""
    df = pd.read_csv(
        data_path,
        header=None,
        sep=",",
        skipinitialspace=True,
        na_values=["?", " ?"],
        engine="python"
    )
    
    print(f"Raw data shape: {df.shape}")
    
    if col_names:
        if len(col_names) != df.shape[1]:
            raise ValueError(
                f"Column mismatch: data has {df.shape[1]} cols, "
                f"columns file has {len(col_names)}"
            )
        df.columns = col_names
    
    return df


def remove_duplicates(df):
    """Remove duplicate rows"""
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")
    
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After deduplication: {df.shape}")
    
    return df


def clean_strings(df):
    """Clean string columns and convert 'Not in universe' to NA"""
    obj_cols = df.select_dtypes(include="object").columns
    
    # Strip whitespace
    df[obj_cols] = df[obj_cols].apply(lambda s: s.astype("string").str.strip())
    
    # Convert "Not in universe" to NA
    def not_in_universe_to_na(x):
        if x is pd.NA or x is None:
            return pd.NA
        if isinstance(x, str) and x.lower().startswith("not in universe"):
            return pd.NA
        return x
    
    for c in obj_cols:
        df[c] = df[c].map(not_in_universe_to_na)
    
    return df


def process_target_variable(df, income_col="label"):
    """Process and binarize the income target variable"""
    # Clean the label column
    df[income_col] = (
        df[income_col]
        .astype("string")
        .str.replace(".", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    
    # Create binary target
    df["income_binary"] = df[income_col].map({"- 50000": 0, "50000+": 1})
    
    print("\nTarget value_counts:")
    print(df[income_col].value_counts(dropna=False))
    
    # Drop rows with invalid target
    before = len(df)
    df = df[df["income_binary"].notna()].reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with invalid/missing target")
    
    return df


def process_weights(df, weight_col="weight"):
    """Process sample weights"""
    if weight_col not in df.columns:
        print(f"Warning: '{weight_col}' column not found. Using uniform weights.")
        df[weight_col] = 1.0
    else:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
        print(f"\nWeight summary:")
        print(df[weight_col].describe())
        print(f"Any non-positive weights? {(df[weight_col] <= 0).any()}")
    
    return df


def drop_sparse_columns(df, cols_to_drop=DROP_SPARSE_COLS):
    """Drop ultra-sparse columns"""
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    print(f"\nDropped sparse columns. Shape: {df.shape}")
    return df


def add_log_transforms(df, cols_to_transform=LOG_TRANSFORM_COLS):
    """Add log-transformed versions of skewed features"""
    for c in cols_to_transform:
        if c in df.columns:
            df[f"{c}__log1p"] = np.log1p(df[c])
    
    log_cols = [c for c in df.columns if c.endswith("__log1p")]
    print(f"\nLog-transformed columns added: {len(log_cols)}")
    return df


def handle_missing_values(df, target_col="income_binary", weight_col="weight"):
    """Handle missing values in numeric and categorical columns"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [target_col, weight_col]]
    
    cat_cols = [c for c in df.columns if c not in num_cols + [target_col, weight_col]]
    
    print(f"\nNumeric columns: {len(num_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")
    
    # Numeric: median imputation
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    
    # Categorical: 'Unknown' placeholder
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")
    
    print(f"Remaining missing values: {df.isna().sum().sum()}")
    
    return df


def preprocess_data(data_path=DATA_PATH, cols_path=COLS_PATH, verbose=True):
    if verbose:
        print("=" * 60)
        print("CENSUS DATA PREPROCESSING PIPELINE")
        print("=" * 60)
    
    # Load data
    col_names = load_column_names(cols_path)
    df = load_raw_data(data_path, col_names)
    
    # Clean and process
    df = remove_duplicates(df)
    df = clean_strings(df)
    df = process_target_variable(df)
    df = process_weights(df)
    
    # Feature engineering
    df = drop_sparse_columns(df)
    df = add_log_transforms(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Convert for sklearn compatibility
    df = df.replace({pd.NA: np.nan})
    
    # Drop raw label column
    df = df.drop(columns=["label"], errors="ignore")
    
    # Prepare modeling matrices
    target_col = "income_binary"
    weight_col = "weight"
    
    X = df.drop(columns=[target_col, weight_col])
    y = df[target_col].astype(int).values
    w = df[weight_col].values
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"PREPROCESSING COMPLETE")
        print(f"Final shape: X={X.shape}, y={y.shape}, w={w.shape}")
        print(f"Target balance: {y.mean():.4f} positive class")
        print("=" * 60)
    
    return df, X, y, w


def get_train_test_split(X, y, w, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate: {y_test.mean():.4f}")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, w_train, w_test


if __name__ == "__main__":
    df, X, y, w = preprocess_data()
    X_train, X_test, y_train, y_test, w_train, w_test = get_train_test_split(X, y, w)
