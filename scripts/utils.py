# Utility functions for model evaluation and visualization

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)


def evaluate_classifier(y_true, y_prob, sample_weight=None, threshold=0.5, label="model"):
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    roc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
    pr = average_precision_score(y_true, y_prob, sample_weight=sample_weight)
    
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", 
        sample_weight=sample_weight, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"{label} @ threshold={threshold:.2f}")
    print(f"{'=' * 60}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nWeighted Confusion Matrix:")
    print(f"           Predicted Neg    Predicted Pos")
    print(f"Actual Neg    {cm[0,0]:>12.0f}    {cm[0,1]:>12.0f}")
    print(f"Actual Pos    {cm[1,0]:>12.0f}    {cm[1,1]:>12.0f}")
    print(f"{'=' * 60}")
    
    return {
        'roc_auc': roc,
        'pr_auc': pr,
        'precision': p,
        'recall': r,
        'f1': f1,
        'confusion_matrix': cm,
        'threshold': threshold
    }


def find_optimal_threshold(y_true, y_prob, sample_weight=None, metric='f1'):
    thresholds = np.linspace(0.05, 0.95, 19)
    best = (0.5, -1)
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary",
            sample_weight=sample_weight, zero_division=0
        )
        
        if metric == 'f1':
            score = f1
        elif metric == 'precision':
            score = p
        elif metric == 'recall':
            score = r
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best[1]:
            best = (t, score)
    
    print(f"\nOptimal threshold: {best[0]:.2f} ({metric}={best[1]:.4f})")
    return best


def get_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_.ravel())
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


def print_model_comparison(results_dict):
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<25} "
              f"{metrics['roc_auc']:>10.4f} "
              f"{metrics['pr_auc']:>10.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f}")
    
    print("=" * 80)


def calculate_weighted_stats(df, group_col, value_col, weight_col):
    def weighted_mean(x, w):
        x_clean = pd.to_numeric(x, errors='coerce')
        mask = np.isfinite(x_clean)
        if mask.sum() == 0:
            return np.nan
        return np.average(x_clean[mask], weights=w[mask])
    
    results = []
    for group in df[group_col].unique():
        mask = df[group_col] == group
        group_df = df[mask]
        w = group_df[weight_col].values
        
        results.append({
            group_col: group,
            'count': len(group_df),
            'weighted_count': w.sum(),
            f'weighted_mean_{value_col}': weighted_mean(group_df[value_col], w)
        })
    
    return pd.DataFrame(results)


def print_segment_profile(segment_df, segment_id, weight_col='weight'):
    print(f"\n{'=' * 60}")
    print(f"SEGMENT {segment_id} PROFILE")
    print(f"{'=' * 60}")
    print(f"Size (unweighted): {len(segment_df):,}")
    print(f"Size (weighted):   {segment_df[weight_col].sum():,.0f}")
    
    # Numeric summary
    numeric_cols = segment_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != weight_col][:5]
    
    if len(numeric_cols) > 0:
        print(f"\nTop Numeric Features:")
        for col in numeric_cols:
            mean_val = np.average(
                segment_df[col].values,
                weights=segment_df[weight_col].values
            )
            print(f"  {col}: {mean_val:.2f}")
    
    print(f"{'=' * 60}")


if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_prob = np.random.rand(1000)
    weights = np.random.rand(1000) * 100
    
    metrics = evaluate_classifier(y_true, y_prob, weights, threshold=0.5, label="Test Model")
    opt_thresh, opt_score = find_optimal_threshold(y_true, y_prob, weights)
