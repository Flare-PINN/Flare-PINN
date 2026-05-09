#!/usr/bin/env python3
"""
Train classical ML baselines (Random Forest, Logistic Regression) on scalar features.

Uses SHARP parameters + derived features for fair comparison with deep learning models.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.svm import SVC
import pickle
import json

# src/baselines/classical/ -> src/baselines/ -> src/ -> project_root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.eval.metrics import sweep_tss, tss_at_threshold, confusion_at_threshold


def load_windows_with_features(windows_path: Path, scalars_path: Path):
    """Load windows and merge with scalar features."""
    windows = pd.read_parquet(windows_path)
    scalars = pd.read_parquet(scalars_path)
    
    # Merge on harpnum and t0
    data = windows.merge(scalars, on=['harpnum', 't0'], how='inner')
    
    print(f"  Loaded {len(data)} windows with scalar features")
    return data


def prepare_features(df: pd.DataFrame, feature_cols: list):
    """Extract features and labels."""
    X = df[feature_cols].values
    
    # Labels for 3 horizons (handle potential _x suffix from merge)
    y_6h_col = 'y_geq_M_6h' if 'y_geq_M_6h' in df.columns else 'y_geq_M_6h_x'
    y_12h_col = 'y_geq_M_12h' if 'y_geq_M_12h' in df.columns else 'y_geq_M_12h_x'
    y_24h_col = 'y_geq_M_24h' if 'y_geq_M_24h' in df.columns else 'y_geq_M_24h_x'
    
    y_6h = df[y_6h_col].values.astype(int)
    y_12h = df[y_12h_col].values.astype(int)
    y_24h = df[y_24h_col].values.astype(int)
    
    harp_nums = df['harpnum'].values
    
    return X, y_6h, y_12h, y_24h, harp_nums


def train_and_evaluate_model(model_name: str, model, X_train, y_train, X_val, y_val, 
                             X_test, y_test, harp_test, horizon: str):
    """Train model and compute metrics."""
    print(f"\n  Training {model_name} for {horizon} horizon...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Debug: Check probability distribution
    print(f"    Prob range: [{test_probs.min():.4f}, {test_probs.max():.4f}], "
          f"mean={test_probs.mean():.4f}, median={np.median(test_probs):.4f}")
    
    # Find optimal threshold on validation set
    tss_val, thresh = sweep_tss(y_val, val_probs, n=1024)
    
    # Evaluate on test set with validation threshold
    tss_test = tss_at_threshold(y_test, test_probs, thresh)
    
    # Confusion matrix
    tp, fp, fn, tn = confusion_at_threshold(y_test, test_probs, thresh)
    
    # Compute metrics
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    
    results = {
        'model': model_name,
        'horizon': horizon,
        'threshold': float(thresh),
        'val_tss': float(tss_val),
        'test_tss': float(tss_test),
        'pod': float(pod),
        'far': float(far),
        'fpr': float(fpr),
        'csi': float(csi),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'n_positives': int(y_test.sum()),
        'n_total': len(y_test),
    }
    
    print(f"    Val TSS: {tss_val:.3f}, Test TSS: {tss_test:.3f}")
    print(f"    Threshold: {thresh:.3f}, POD: {pod:.3f}, FAR: {far:.3f}")
    
    return results, model


def main():
    print("=" * 80)
    print("TRAINING CLASSICAL ML BASELINES")
    print("=" * 80)
    
    # Paths
    scalars_path = project_root / "data/interim/scalar_features.parquet"
    train_val_path = project_root / "data/interim/windows_train_val_8005.parquet"
    test_path = project_root / "data/interim/windows_test_15.parquet"
    
    # Model outputs go to outputs/, final results CSV goes to final_results/
    model_output_dir = project_root / "outputs/classical_baselines"
    results_output_dir = project_root / "final_results/classical_baselines"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check files exist
    for path in [scalars_path, train_val_path, test_path]:
        if not path.exists():
            print(f" Missing: {path}")
            return
    
    print("\n📂 Loading data...")
    train_val_data = load_windows_with_features(train_val_path, scalars_path)
    test_data = load_windows_with_features(test_path, scalars_path)
    
    # Split train/val (5% validation)
    n_val = int(len(train_val_data) * 0.0588)
    train_data = train_val_data.iloc[:-n_val]
    val_data = train_val_data.iloc[-n_val:]
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Feature columns (derived features computed by the data pipeline)
    # These are: r_value, gwpil, obs_coverage, frame_count + 8 PIL evolution + 4 temporal stats
    feature_cols = [f'feat_{i:02d}' for i in range(16)]
    
    # Check which features are available
    available_features = [f for f in feature_cols if f in train_data.columns]
    print(f"\n  Using {len(available_features)} derived features")
    
    if len(available_features) == 0:
        print(" No features found in scalar_features.parquet")
        print("   Available columns:", [c for c in train_data.columns if 'feat' in c])
        return
    
    # Prepare data
    print("\n Preparing features...")
    X_train, y_train_6h, y_train_12h, y_train_24h, _ = prepare_features(train_data, available_features)
    X_val, y_val_6h, y_val_12h, y_val_24h, _ = prepare_features(val_data, available_features)
    X_test, y_test_6h, y_test_12h, y_test_24h, harp_test = prepare_features(test_data, available_features)
    
    # Handle missing values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"  Train positives: 6h={y_train_6h.sum()}, 12h={y_train_12h.sum()}, 24h={y_train_24h.sum()}")
    print(f"  Test positives:  6h={y_test_6h.sum()}, 12h={y_test_12h.sum()}, 24h={y_test_24h.sum()}")
    
    # Models to train
    # NOTE: Random Forest and Gradient Boosting perform poorly on these highly engineered  
    # features with severe class imbalance. XGBoost handles imbalanced data much better.
    
    # Compute scale_pos_weight for XGBoost (ratio of negative to positive samples)
    scale_pos_weight_6h = (len(train_data) - y_train_6h.sum()) / (y_train_6h.sum() + 1)
    scale_pos_weight_12h = (len(train_data) - y_train_12h.sum()) / (y_train_12h.sum() + 1)
    scale_pos_weight_24h = (len(train_data) - y_train_24h.sum()) / (y_train_24h.sum() + 1)
    
    models_by_horizon = {
        '6h': {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                scale_pos_weight=scale_pos_weight_6h,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'Logistic Regression': LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            'SVM': SGDClassifier(
                loss='modified_huber',  # SVM-like but supports probabilities
                penalty='l2',
                alpha=0.0001,
                class_weight='balanced',
                max_iter=10000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1
            ),
        },
        '12h': {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                scale_pos_weight=scale_pos_weight_12h,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'Logistic Regression': LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            'SVM': SGDClassifier(
                loss='modified_huber',  # SVM-like but supports probabilities
                penalty='l2',
                alpha=0.0001,
                class_weight='balanced',
                max_iter=10000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1
            ),
        },
        '24h': {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                scale_pos_weight=scale_pos_weight_24h,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'Logistic Regression': LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            'SVM': SGDClassifier(
                loss='modified_huber',  # SVM-like but supports probabilities
                penalty='l2',
                alpha=0.0001,
                class_weight='balanced',
                max_iter=10000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1
            ),
        },
    }
    
    all_results = []
    trained_models = {}
    
    # Train for each horizon
    for horizon_name, y_train, y_val, y_test in [
        ('6h', y_train_6h, y_val_6h, y_test_6h),
        ('12h', y_train_12h, y_val_12h, y_test_12h),
        ('24h', y_train_24h, y_val_24h, y_test_24h),
    ]:
        print(f"\n{'=' * 80}")
        print(f"HORIZON: {horizon_name}")
        print('=' * 80)
        
        models = models_by_horizon[horizon_name]
        
        for model_name, model in models.items():
            results, trained_model = train_and_evaluate_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                harp_test=harp_test,
                horizon=horizon_name
            )
            
            all_results.append(results)
            trained_models[f"{model_name}_{horizon_name}"] = trained_model
    
    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print('=' * 80)
    
    # Save final results CSV to final_results/
    results_df = pd.DataFrame(all_results)
    results_path = results_output_dir / "classical_baselines_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f" Results CSV saved to: {results_path}")
    
    # Save models to outputs/
    for name, model in trained_models.items():
        model_path = model_output_dir / f"{name.replace(' ', '_').lower()}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    print(f" Models saved to: {model_output_dir}")
    
    # Save scaler to outputs/
    scaler_path = model_output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata to outputs/
    metadata = {
        'features': available_features,
        'n_features': len(available_features),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
    }
    metadata_path = model_output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: 24h HORIZON COMPARISON")
    print('=' * 80)
    
    results_24h = results_df[results_df['horizon'] == '24h'].sort_values('test_tss', ascending=False)
    
    print(f"\n{'Model':<25} {'Test TSS':<12} {'POD':<8} {'FAR':<8}")
    print('-' * 80)
    for _, row in results_24h.iterrows():
        print(f"{row['model']:<25} {row['test_tss']:<12.3f} {row['pod']:<8.3f} {row['far']:<8.3f}")
    
    print(f"\n All baselines trained and evaluated!")
    print(f"   Results: {results_path}")


if __name__ == "__main__":
    main()
