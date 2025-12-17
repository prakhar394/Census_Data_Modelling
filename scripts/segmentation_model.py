# Customer Segmentation Model

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import *
from data_preprocessing import preprocess_data


def prepare_segmentation_data(df, target_col='income_binary', weight_col='weight'):
    if 'age' not in df.columns:
        raise ValueError("'age' column not found in DataFrame")
    
    df_adult = df[df['age'] >= 18].reset_index(drop=True)
    print(f"\nAdult-only dataframe shape: {df_adult.shape}")
    
    if weight_col in df_adult.columns:
        weights = df_adult[weight_col].fillna(1.0).values
    else:
        weights = np.ones(len(df_adult))
    
    y_income = df_adult[target_col].astype(int).values if target_col in df_adult.columns else None
    
    drop_cols = []
    if target_col in df_adult.columns:
        drop_cols.append(target_col)
    if weight_col in df_adult.columns:
        drop_cols.append(weight_col)
    if 'label' in df_adult.columns:
        drop_cols.append('label')
    
    X_seg = df_adult.drop(columns=drop_cols, errors='ignore')
    
    print(f"Segmentation features: {X_seg.shape[1]} columns")
    
    return X_seg, y_income, weights, df_adult


def create_segmentation_preprocessor(X_seg):
    num_cols = X_seg.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_seg.columns if c not in num_cols]
    
    print(f"\nSegmentation preprocessing:")
    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            min_frequency=50,
            sparse_output=True
        ))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, num_cols),
            ('cat', categorical_pipe, cat_cols)
        ],
        remainder='drop',
        sparse_threshold=0.3
    )
    
    return preprocessor


def find_optimal_k(X_embedded, weights, k_range=None, sample_size=None):
    if k_range is None:
        k_range = range(
            SEGMENTATION_PARAMS['k_min'],
            SEGMENTATION_PARAMS['k_max'] + 1
        )
    
    if sample_size is None:
        sample_size = min(
            SEGMENTATION_PARAMS['sample_for_silhouette'],
            X_embedded.shape[0]
        )
    
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(X_embedded.shape[0], size=sample_size, replace=False)
    
    X_sample = X_embedded[idx]
    w_sample = weights[idx]
    
    print(f"\nFinding optimal K using {sample_size} samples...")
    print(f"Testing K values: {list(k_range)}")
    
    silhouette_scores = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=RANDOM_STATE)
        kmeans.fit(X_sample, sample_weight=w_sample)
        
        score = silhouette_score(X_sample, kmeans.labels_, metric='euclidean')
        silhouette_scores[k] = score
        print(f"  K={k}: silhouette={score:.4f}")
    
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\nOptimal K: {optimal_k} (silhouette={silhouette_scores[optimal_k]:.4f})")
    
    return optimal_k, silhouette_scores


def create_segments(X_embedded, weights, n_clusters):
    print(f"\n" + "=" * 60)
    print(f"CREATING {n_clusters} CUSTOMER SEGMENTS")
    print("=" * 60)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=RANDOM_STATE)
    kmeans.fit(X_embedded, sample_weight=weights)
    
    segment_labels = kmeans.labels_
    
    print(f"Segmentation completed")
    print(f"Segment distribution:")
    for i in range(n_clusters):
        count = (segment_labels == i).sum()
        weighted_count = weights[segment_labels == i].sum()
        print(f"  Segment {i}: {count:,} samples ({weighted_count:,.0f} weighted)")
    
    return kmeans, segment_labels


def profile_segments(df_adult, segment_labels, weights, y_income=None):
    df_segments = df_adult.copy()
    df_segments['segment'] = segment_labels
    
    def wmean(x, w):
        x_clean = pd.to_numeric(x, errors='coerce')
        mask = np.isfinite(x_clean)
        if mask.sum() == 0:
            return np.nan
        return np.average(x_clean[mask], weights=w[mask])
    
    def top_cats(series, w, topn=3):
        tmp = pd.DataFrame({'v': series.astype('string'), 'w': w})
        tmp = tmp[tmp['v'].notna()]
        if tmp.empty:
            return ""
        vc = tmp.groupby('v')['w'].sum().sort_values(ascending=False).head(topn)
        return " | ".join(vc.index)
    
    profiles = []
    
    for seg_id in sorted(df_segments['segment'].unique()):
        seg_df = df_segments[df_segments['segment'] == seg_id]
        seg_weights = weights[segment_labels == seg_id]
        
        profile = {
            'segment': seg_id,
            'n_unweighted': len(seg_df),
            'n_weighted': seg_weights.sum()
        }
        
        if y_income is not None:
            seg_income = y_income[segment_labels == seg_id]
            profile['income_gt_50k_rate'] = np.average(seg_income, weights=seg_weights)
        
        for col in SEGMENTATION_NUMERIC_FOCUS:
            if col in seg_df.columns:
                profile[f'mean_{col}'] = wmean(seg_df[col], seg_weights)
        
        for col in SEGMENTATION_CATEGORICAL_FOCUS:
            if col in seg_df.columns:
                profile[f'top_{col}'] = top_cats(seg_df[col], seg_weights)
        
        profiles.append(profile)
    
    profile_df = pd.DataFrame(profiles)
    
    if 'income_gt_50k_rate' in profile_df.columns:
        profile_df = profile_df.sort_values('income_gt_50k_rate', ascending=False)
    
    return profile_df


def print_segment_profiles(profile_df):
    print("\n" + "=" * 80)
    print("CUSTOMER SEGMENT PROFILES")
    print("=" * 80)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 60)
    
    print(profile_df.to_string(index=False))
    print("=" * 80)


def save_segmentation_model(preprocessor, svd, kmeans, output_dir=MODELS_DIR):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    components = {
        'preprocessor': preprocessor,
        'svd': svd,
        'kmeans': kmeans
    }
    
    filename = output_path / 'segmentation_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(components, f)
    
    print(f"\nSegmentation model saved to {filename}")


def main():
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)
    
    df, _, _, _ = preprocess_data()
    X_seg, y_income, weights, df_adult = prepare_segmentation_data(df)
    preprocessor = create_segmentation_preprocessor(X_seg)
    
    print("\nApplying preprocessing...")
    X_sparse = preprocessor.fit_transform(X_seg)
    print(f"Preprocessed shape (sparse): {X_sparse.shape}")
    
    print("\nApplying dimensionality reduction...")
    n_components = SEGMENTATION_PARAMS['svd_components']
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_embedded = svd.fit_transform(X_sparse)
    print(f"Embedding shape: {X_embedded.shape}")
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    
    optimal_k, silhouette_scores = find_optimal_k(X_embedded, weights)
    kmeans, segment_labels = create_segments(X_embedded, weights, optimal_k)
    profile_df = profile_segments(df_adult, segment_labels, weights, y_income)
    print_segment_profiles(profile_df)
    
    save_segmentation_model(preprocessor, svd, kmeans)
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df_adult['segment'] = segment_labels
    df_adult.to_csv(output_path / 'segmented_customers.csv', index=False)
    print(f"\nSegment assignments saved to {output_path / 'segmented_customers.csv'}")
    
    profile_df.to_csv(output_path / 'segment_profiles.csv', index=False)
    print(f"Segment profiles saved to {output_path / 'segment_profiles.csv'}")
    
    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETE")
    print("=" * 60)
    
    return profile_df, segment_labels


if __name__ == "__main__":
    profile_df, segments = main()
