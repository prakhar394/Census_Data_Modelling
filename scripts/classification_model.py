# Classification Model Training and Evaluation

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available")

from config import *
from data_preprocessing import preprocess_data, get_train_test_split
from utils import evaluate_classifier, find_optimal_threshold, print_model_comparison


def create_preprocessing_pipeline(X_train):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    
    print(f"\nPreprocessing pipeline:")
    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor


def train_logistic_regression(X_train, y_train, w_train, preprocessor):
    """Train Logistic Regression model"""
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)
    
    model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    
    print("Logistic Regression training completed")
    return pipeline


def train_random_forest(X_train, y_train, w_train, preprocessor):
    """Train Random Forest model"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)
    
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    
    print("Random Forest training completed")
    return pipeline


def train_xgboost(X_train, y_train, w_train, preprocessor):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available, skipping...")
        return None
    
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)
    
    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()
    
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    model = xgb.XGBClassifier(**params)
    
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    
    print("XGBoost training completed")
    return pipeline


def train_lightgbm(X_train, y_train, w_train, preprocessor):
    """Train LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available, skipping...")
        return None
    
    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM")
    print("=" * 60)
    
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    
    print("LightGBM training completed")
    return pipeline


def evaluate_all_models(models_dict, X_test, y_test, w_test):
    results = {}
    
    for name, model in models_dict.items():
        if model is None:
            continue
        
        print(f"\nEvaluating {name}...")
        
        y_prob = model.predict_proba(X_test)[:, 1]
        opt_thresh, _ = find_optimal_threshold(y_test, y_prob, w_test, metric='f1')
        metrics = evaluate_classifier(
            y_test, y_prob, w_test,
            threshold=opt_thresh,
            label=name
        )
        
        results[name] = metrics
    
    if len(results) > 0:
        print_model_comparison(results)
    
    return results


def save_models(models_dict, output_dir=MODELS_DIR):
    """Save trained models to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, model in models_dict.items():
        if model is None:
            continue
        
        filename = output_path / f"{name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} to {filename}")


def load_model(model_name, models_dir=MODELS_DIR):
    """Load a saved model from disk"""
    filename = Path(models_dir) / f"{model_name.lower().replace(' ', '_')}.pkl"
    
    if not filename.exists():
        raise FileNotFoundError(f"Model file not found: {filename}")
    
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded model from {filename}")
    return model


def get_feature_names_from_preprocessor(preprocessor, X_original):
    """Extract feature names after preprocessing pipeline"""
    try:
        # Try to get feature names directly from the preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            return preprocessor.get_feature_names_out(X_original.columns)
    except:
        pass
    
    # Manual extraction
    num_cols = X_original.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_original.columns if c not in num_cols]
    
    feature_names = []
    
    # Add numeric features
    feature_names.extend(num_cols)
    
    # Add categorical features (one-hot encoded)
    if len(cat_cols) > 0:
        try:
            cat_transformer = preprocessor.named_transformers_['cat']
            ohe = cat_transformer.named_steps['onehot']
            
            # Get one-hot encoded feature names
            if hasattr(ohe, 'get_feature_names_out'):
                cat_feature_names = ohe.get_feature_names_out(cat_cols)
                feature_names.extend(cat_feature_names)
            else:
                # Fallback: transform a sample to get number of features
                from scipy.sparse import issparse
                sample_transform = cat_transformer.transform(X_original[cat_cols].iloc[:1])
                if issparse(sample_transform):
                    n_cat_features = sample_transform.shape[1]
                else:
                    n_cat_features = sample_transform.shape[1]
                for col in cat_cols:
                    for i in range(n_cat_features // len(cat_cols)):
                        feature_names.append(f"{col}_cat_{i}")
        except Exception as e:
            # Final fallback
            print(f"  Warning: Could not extract categorical feature names: {e}")
            # Estimate number of categorical features
            sample_transform = preprocessor.transform(X_original.iloc[:1])
            from scipy.sparse import issparse
            if issparse(sample_transform):
                n_total = sample_transform.shape[1]
            else:
                n_total = sample_transform.shape[1]
            n_cat_features = n_total - len(num_cols)
            for i in range(n_cat_features):
                feature_names.append(f"cat_feature_{i}")
    
    return feature_names


def generate_shap_plots(model_pipeline, X_train, X_test, model_name, output_dir=OUTPUT_DIR, sample_size=1000):
    """Generate and save SHAP plots for a model"""
    if not SHAP_AVAILABLE:
        print(f"SHAP not available, skipping plots for {model_name}")
        return
    
    print(f"\nGenerating SHAP plots for {model_name}...")
    
    # Extract preprocessor and model from pipeline
    preprocessor = model_pipeline.named_steps['preprocess']
    model = model_pipeline.named_steps['model']
    
    # Transform data (convert sparse matrices to dense if needed)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Convert sparse matrices to dense arrays if needed
    from scipy.sparse import issparse
    if issparse(X_train_transformed):
        X_train_transformed = X_train_transformed.toarray()
    if issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()
    
    # Sample data for SHAP (to speed up computation)
    n_samples = min(sample_size, X_test_transformed.shape[0])
    np.random.seed(RANDOM_STATE)  # For reproducibility
    sample_idx = np.random.choice(X_test_transformed.shape[0], n_samples, replace=False)
    X_test_sample = X_test_transformed[sample_idx]
    
    # Get feature names
    try:
        feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)
        # Ensure feature names match the actual transformed data shape
        if len(feature_names) != X_train_transformed.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]
    except Exception as e:
        print(f"  Warning: Could not extract feature names: {e}")
        feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]
    
    # Sample background data for explainers that need it
    background_size = min(100, X_train_transformed.shape[0])
    background_idx = np.random.choice(X_train_transformed.shape[0], background_size, replace=False)
    background_data = X_train_transformed[background_idx]
    
    # Select appropriate SHAP explainer based on model type
    model_type = type(model).__name__
    
    try:
        if model_type in ['XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier']:
            # Tree-based models - use TreeExplainer (fast and exact)
            # TreeExplainer doesn't require background data, but can use it
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            
            # For binary classification, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1 (positive class)
        elif model_type == 'LogisticRegression':
            # Linear models - use LinearExplainer (requires background)
            explainer = shap.LinearExplainer(model, background_data)
            shap_values = explainer.shap_values(X_test_sample)
        else:
            # Fallback to KernelExplainer (slower but works for any model)
            print(f"  Using KernelExplainer for {model_type} (this may be slow)...")
            explainer = shap.KernelExplainer(model.predict_proba, background_data[:50])
            shap_values = explainer.shap_values(X_test_sample[:100])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Create output directory for SHAP plots
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        shap_dir = output_path / "shap_plots"
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        X_test_df = pd.DataFrame(X_test_sample, columns=feature_names[:X_test_sample.shape[1]])
        
        # Limit feature names to match actual data
        if len(feature_names) > X_test_sample.shape[1]:
            feature_names = feature_names[:X_test_sample.shape[1]]
        
        # 1. Summary plot (beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, show=False, max_display=20)
        plt.tight_layout()
        summary_path = shap_dir / f"{model_name.lower().replace(' ', '_')}_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved summary plot: {summary_path}")
        
        # 2. Summary plot (bar - mean absolute SHAP values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        bar_path = shap_dir / f"{model_name.lower().replace(' ', '_')}_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved bar plot: {bar_path}")
        
        # 3. Waterfall plot for first prediction (if supported)
        try:
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=X_test_df.iloc[0],
                    feature_names=feature_names
                ),
                show=False,
                max_display=15
            )
            plt.tight_layout()
            waterfall_path = shap_dir / f"{model_name.lower().replace(' ', '_')}_waterfall.png"
            plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved waterfall plot: {waterfall_path}")
        except Exception as e:
            print(f"  Could not generate waterfall plot: {e}")
        
        # 4. Dependence plots for top 4 features
        try:
            # Get top features by mean absolute SHAP value
            mean_abs_shap = np.abs(shap_values).mean(0)
            top_features_idx = np.argsort(mean_abs_shap)[-4:][::-1]
            top_features = [feature_names[i] for i in top_features_idx]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
            
            for idx, (ax, feat_idx) in enumerate(zip(axes, top_features_idx)):
                try:
                    shap.dependence_plot(
                        feat_idx,
                        shap_values,
                        X_test_df,
                        show=False,
                        ax=ax
                    )
                    ax.set_title(f"Dependence Plot: {feature_names[feat_idx]}")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Could not plot {feature_names[feat_idx]}", 
                           ha='center', va='center', transform=ax.transAxes)
                    print(f"  Warning: Could not create dependence plot for {feature_names[feat_idx]}: {e}")
            
            plt.tight_layout()
            dependence_path = shap_dir / f"{model_name.lower().replace(' ', '_')}_dependence.png"
            plt.savefig(dependence_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved dependence plots: {dependence_path}")
        except Exception as e:
            print(f"  Could not generate dependence plots: {e}")
        
        print(f"  SHAP plots for {model_name} completed successfully")
        
    except Exception as e:
        print(f"  Error generating SHAP plots for {model_name}: {e}")
        import traceback
        traceback.print_exc()


def generate_rf_feature_importance(model_pipeline, X_train, output_dir=OUTPUT_DIR, top_n=20):
    """Generate feature importance plot for Random Forest model"""
    print(f"\nGenerating feature importance plot for Random Forest...")
    
    try:
        # Extract preprocessor and model from pipeline
        preprocessor = model_pipeline.named_steps['preprocess']
        model = model_pipeline.named_steps['model']
        
        # Transform training data to get actual feature count
        X_train_transformed = preprocessor.transform(X_train.iloc[:1])
        from scipy.sparse import issparse
        if issparse(X_train_transformed):
            n_features = X_train_transformed.shape[1]
        else:
            n_features = X_train_transformed.shape[1]
        
        # Get feature names
        try:
            feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)
            # Ensure we have the right number of features
            if len(feature_names) != n_features:
                feature_names = [f"feature_{i}" for i in range(n_features)]
        except:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Ensure feature names match importances
        if len(feature_names) != len(model.feature_importances_):
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Random Forest - Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        shap_dir = output_path / "shap_plots"
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        importance_path = shap_dir / "random_forest_feature_importance.png"
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved feature importance plot: {importance_path}")
        
    except Exception as e:
        print(f"  Error generating feature importance plot: {e}")
        import traceback
        traceback.print_exc()


def generate_shap_plots_for_all_models(models_dict, X_train, X_test, output_dir=OUTPUT_DIR):
    """Generate SHAP plots for XGBoost and LightGBM, feature importance for Random Forest"""
    print("\n" + "=" * 60)
    print("GENERATING INTERPRETABILITY PLOTS")
    print("=" * 60)
    
    # Generate feature importance for Random Forest
    if 'Random Forest' in models_dict and models_dict['Random Forest'] is not None:
        generate_rf_feature_importance(models_dict['Random Forest'], X_train, output_dir)
    
    # Generate SHAP plots for XGBoost and LightGBM only
    if not SHAP_AVAILABLE:
        print("\nSHAP not available, skipping SHAP plot generation")
        return
    
    models_for_shap = ['XGBoost', 'LightGBM']
    
    for name in models_for_shap:
        if name in models_dict and models_dict[name] is not None:
            generate_shap_plots(models_dict[name], X_train, X_test, name, output_dir)


def main():
    print("\n" + "=" * 60)
    print("CENSUS INCOME CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    df, X, y, w = preprocess_data()
    X_train, X_test, y_train, y_test, w_train, w_test = get_train_test_split(X, y, w)
    
    preprocessor = create_preprocessing_pipeline(X_train)
    
    models = {}
    models['Logistic Regression'] = train_logistic_regression(
        X_train, y_train, w_train, preprocessor
    )
    models['Random Forest'] = train_random_forest(
        X_train, y_train, w_train, preprocessor
    )
    models['XGBoost'] = train_xgboost(
        X_train, y_train, w_train, preprocessor
    )
    models['LightGBM'] = train_lightgbm(
        X_train, y_train, w_train, preprocessor
    )
    
    results = evaluate_all_models(models, X_test, y_test, w_test)
    save_models(models)
    
    # Generate SHAP plots for all models
    generate_shap_plots_for_all_models(models, X_train, X_test, OUTPUT_DIR)
    
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        print(f"\n" + "=" * 60)
        print(f"BEST MODEL: {best_model_name}")
        print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        print(f"PR-AUC: {results[best_model_name]['pr_auc']:.4f}")
        print("=" * 60)
    
    print("\nClassification pipeline completed successfully!")
    return models, results


if __name__ == "__main__":
    models, results = main()
