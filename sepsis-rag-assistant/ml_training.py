import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

def default_feature_selector(df, target_col):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != target_col]

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
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        print("Please ensure your data generation script has been run.")
        return
    except Exception as e:
        print(f"Failed to load the CSV file. Error: {e}")
        return

    print("[+] Data shape:", df.shape)
    print("[+] Columns:", df.columns.tolist())

    if features is None:
        features = default_feature_selector(df, target_col)
        print(f"[+] Auto-selected {len(features)} numeric features")

    # If 'Consciousness Level' is a feature, we need to handle it.
    categorical_features = ['Consciousness Level'] if 'Consciousness Level' in df.columns else []
    
    # Quick strategy: impute numeric features with column mean
    X = df[features + categorical_features].copy()
    y = df[target_col].copy()

    # Preprocess categorical features
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Split BEFORE scaling/imputing to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
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

    # Check class balance
    print("[+] Class distribution train:", y_train.value_counts(normalize=True).to_dict())

    # Model training: RandomForestClassifier with optional grid search
    print("[+] Training RandomForestClassifier")
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    if run_grid_search:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 2, 4]
        }
        gs = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
        gs.fit(X_train_scaled, y_train)
        best = gs.best_estimator_
        print("[+] GridSearch best params:", gs.best_params_)
    else:
        best = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
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
    model_path = Path(out_dir) / "rf_model.joblib"
    scaler_path = Path(out_dir) / "scaler.joblib"
    imputer_path = Path(out_dir) / "imputer.joblib"
    features_path = Path(out_dir) / "features.json"

    joblib.dump(best, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump({"features": X.columns.tolist()}, fh, indent=2)

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
    parser.add_argument("--no_grid", action="store_true", help="Skip GridSearch")
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
