"""
Telco Customer Churn - Module 4/5: Model Building & Training

Business goal
- Train multiple baseline models for churn prediction and compare performance later
- Ensure reproducibility: fixed random seed, stratified split, saved artifacts

Dependencies
pip install pandas numpy scikit-learn joblib
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
SCRIPT_PREFIX = "step4_train_models"
UPSTREAM_PREFIX = "step3_feature_engineering"


def ensure_dirs() -> dict[str, Path]:
    base = Path(__file__).resolve().parent
    outputs = base / "outputs"
    csv_dir = outputs / "csv"
    reports_dir = outputs / "reports"
    models_dir = outputs / "models"

    outputs.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base, "outputs": outputs, "csv": csv_dir, "reports": reports_dir, "models": models_dir}


def resolve_feature_paths(x_path: str | None, y_path: str | None) -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent

    candidates_x = [Path(x_path)] if x_path else []
    candidates_y = [Path(y_path)] if y_path else []

    candidates_x.extend(
        [
            here / "outputs" / "csv" / f"{UPSTREAM_PREFIX}__features_X.csv",
            here / "outputs" / f"{UPSTREAM_PREFIX}__features_X.csv",
        ]
    )
    candidates_y.extend(
        [
            here / "outputs" / "csv" / f"{UPSTREAM_PREFIX}__labels_y.csv",
            here / "outputs" / f"{UPSTREAM_PREFIX}__labels_y.csv",
        ]
    )

    Xp = next((p for p in candidates_x if p.exists() and p.is_file()), None)
    yp = next((p for p in candidates_y if p.exists() and p.is_file()), None)
    if not Xp or not yp:
        raise FileNotFoundError("Feature/label CSV not found. Run step3_feature_engineering.py first, or pass --x/--y.")

    return Xp, yp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - train models")
    parser.add_argument("--x", type=str, default=None, help="Feature CSV path (optional override)")
    parser.add_argument("--y", type=str, default=None, help="Label CSV path (optional override)")
    args = parser.parse_args()

    dirs = ensure_dirs()
    X_path, y_path = resolve_feature_paths(args.x, args.y)

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)["ChurnLabel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)

    models = {"logistic_regression": lr, "decision_tree": dt, "random_forest": rf}
    for m in models.values():
        m.fit(X_train, y_train)

    saved_models = {}
    for name, model in models.items():
        p = dirs["models"] / f"{SCRIPT_PREFIX}__{name}.joblib"
        joblib.dump(model, p)
        saved_models[name] = str(p)

    X_train_path = dirs["csv"] / f"{SCRIPT_PREFIX}__X_train.csv"
    X_test_path = dirs["csv"] / f"{SCRIPT_PREFIX}__X_test.csv"
    y_train_path = dirs["csv"] / f"{SCRIPT_PREFIX}__y_train.csv"
    y_test_path = dirs["csv"] / f"{SCRIPT_PREFIX}__y_test.csv"
    X_train.to_csv(X_train_path, index=False, encoding="utf-8")
    X_test.to_csv(X_test_path, index=False, encoding="utf-8")
    y_train.to_frame("ChurnLabel").to_csv(y_train_path, index=False, encoding="utf-8")
    y_test.to_frame("ChurnLabel").to_csv(y_test_path, index=False, encoding="utf-8")

    summary_path = dirs["reports"] / f"{SCRIPT_PREFIX}__training_summary.txt"
    lines: list[str] = []
    lines.append("Telco Customer Churn - Model Training Summary")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("1) Train/test split")
    lines.append("- Split ratio: 70/30 (train/test)")
    lines.append(f"- Random seed: {RANDOM_STATE}")
    lines.append("- Stratified by churn label to keep class proportions consistent.")
    lines.append(f"- Train size: {len(X_train)}; Test size: {len(X_test)}")
    lines.append("")
    lines.append("2) Models")
    lines.append("- Logistic Regression: interpretable baseline; churn probability supports risk ranking.")
    lines.append(f"  Params: {lr.get_params()}")
    lines.append("- Decision Tree: rule-based interpretability; can overfit, used mainly for comparison.")
    lines.append(f"  Params: {dt.get_params()}")
    lines.append("- Random Forest: robust ensemble; captures non-linearities and interactions well.")
    lines.append(f"  Params: {rf.get_params()}")
    lines.append("")
    lines.append("3) Saved artifacts")
    lines.append(f"- Models directory: {dirs['models']}")
    for k, v in saved_models.items():
        lines.append(f"  - {k}: {v}")
    lines.append("- Split datasets (CSV):")
    lines.append(f"  - X_train: {X_train_path}")
    lines.append(f"  - X_test: {X_test_path}")
    lines.append(f"  - y_train: {y_train_path}")
    lines.append(f"  - y_test: {y_test_path}")

    summary_path.write_text("\n".join(lines), encoding="utf-8", errors="replace")

    print("=== Training complete ===")
    print(f"Saved models to: {dirs['models']}")
    print("Saved split CSVs to outputs/csv/:")
    print(f"- {X_train_path}")
    print(f"- {X_test_path}")
    print(f"- {y_train_path}")
    print(f"- {y_test_path}")
    print(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()

