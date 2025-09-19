#!/usr/bin/env python3
"""
ml_training.py

Train a KNeighborsClassifier on a sepsis dataset CSV and save:
 - models/knn_model.joblib
 - models/scaler.joblib
 - models/features.json

Usage:
    python ml_training.py --data data/sepsis_kaggle.csv --target sepsis_label \
                         --out_dir models --test_size 0.2 --random_state 42
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

def default_feature_selector(df, target_col):
    # Heuristic: all numeric columns except target are features
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c != target_col]

def main(
    csv_path: str,
    target_col: str,
    features: list | None,
    out_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    run_grid_search: bool = True
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print("[+] Data shape:", df.shape)
    print("[+] Columns:", df.columns.tolist())

    if features is None:
        features = default_feature_selector(df, target_col)
        print(f"[+] Auto-selected {len(features)} numeric features")

    # Basic EDA: missing values
    missing = df[features + [target_col]].isna().mean().sort_values(ascending=False)
    print("[+] Missing rates (top 20):")
    print(missing.head(20))

    # Quick strategy: impute numeric features with column mean
    X = df[features].copy()
    y = df[target_col].copy()

    # If the target is not numeric (e.g., 'yes'/'no'), map to 0/1
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).map(lambda v: 1 if str(v).lower() in ("1","yes","true","y","sepsis") else 0)

    # Split BEFORE scaling/imputing to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()>1 else None
    )
    print("[+] Train/test split:", X_train.shape, X_test.shape)

    # Imputer (fit on train only)
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Scaler (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Simple class balance check
    print("[+] Class distribution train:", y_train.value_counts(normalize=True).to_dict())

    # Model training: KNN with optional grid search for n_neighbors
    print("[+] Training KNeighborsClassifier")
    knn = KNeighborsClassifier()
    if run_grid_search:
        param_grid = {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
        gs = GridSearchCV(knn, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
        gs.fit(X_train_scaled, y_train)
        best = gs.best_estimator_
        print("[+] GridSearch best params:", gs.best_params_)
    else:
        best = KNeighborsClassifier(n_neighbors=5)
        best.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = best.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, digits=4)
    acc = accuracy_score(y_test, y_pred)
    try:
        y_score = best.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = None

    print("[+] Test accuracy:", acc)
    if auc is not None:
        print("[+] Test ROC AUC:", auc)
    print("[+] Classification report:\n", report)

    # Save artifacts: model, scaler, imputer, features list
    model_path = Path(out_dir) / "knn_model.joblib"
    scaler_path = Path(out_dir) / "scaler.joblib"
    imputer_path = Path(out_dir) / "imputer.joblib"
    features_path = Path(out_dir) / "features.json"

    joblib.dump(best, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump({"features": features}, fh, indent=2)

    print(f"[+] Saved model -> {model_path}")
    print(f"[+] Saved scaler -> {scaler_path}")
    print(f"[+] Saved imputer -> {imputer_path}")
    print(f"[+] Saved features manifest -> {features_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "--csv", dest="csv_path", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", dest="target_col", required=True, help="Target column name (0/1)")
    parser.add_argument("--features", dest="features", nargs="+", default=None, help="List of feature column names (optional)")
    parser.add_argument("--out_dir", dest="out_dir", default="models", help="Where to write model/scaler")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_grid", action="store_true", help="Skip GridSearch (use default KNN)")
    args = parser.parse_args()

    main(
        args.csv_path,
        args.target_col,
        args.features,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        run_grid_search=not args.no_grid
    )
