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

from project_utils import Timer, ensure_output_dirs, get_paths, load_config, setup_logger


RANDOM_STATE = 42
SCRIPT_PREFIX = "step4_train_models"
UPSTREAM_PREFIX = "step3_feature_engineering"


def resolve_config_path(cli_config: str | None) -> Path:
    base = Path(__file__).resolve().parent
    return Path(cli_config).resolve() if cli_config else (base / "config.yaml")


def resolve_feature_paths(x_path: str | None, y_path: str | None) -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent

    candidates_x = [Path(x_path)] if x_path else []
    candidates_y = [Path(y_path)] if y_path else []

    candidates_x.extend(
        [
            here / "outputs" / "csv" / "step3__engineered_features.csv",
            here / "outputs" / "csv" / f"{UPSTREAM_PREFIX}__features_X.csv",
            here / "outputs" / f"{UPSTREAM_PREFIX}__features_X.csv",
        ]
    )
    candidates_y.extend(
        [
            here / "outputs" / "csv" / "step3__engineered_features.csv",  # contains ChurnLabel too
            here / "outputs" / "csv" / f"{UPSTREAM_PREFIX}__labels_y.csv",
            here / "outputs" / f"{UPSTREAM_PREFIX}__labels_y.csv",
        ]
    )

    Xp = next((p for p in candidates_x if p.exists() and p.is_file()), None)
    yp = next((p for p in candidates_y if p.exists() and p.is_file()), None)
    if not Xp or not yp:
        raise FileNotFoundError("Feature/label CSV not found. Run step3_feature_engineering.py first, or pass --x/--y.")

    return Xp, yp


def train_and_split(
    X: pd.DataFrame,
    y: pd.Series,
    training_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    """
    Split data (stratified) and train the 3 baseline models.

    Parameters
    - X: feature dataframe
    - y: label series (0/1)
    - training_cfg: config["training"] mapping

    Returns
    - X_train, X_test, y_train, y_test
    - trained models dict
    """

    test_size = float(training_cfg.get("test_size", 0.3))
    random_state = int(training_cfg.get("random_state", RANDOM_STATE))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    models_cfg = training_cfg.get("models", {}) if isinstance(training_cfg.get("models"), dict) else {}
    lr_cfg = models_cfg.get("logistic_regression", {}) if isinstance(models_cfg.get("logistic_regression"), dict) else {}
    dt_cfg = models_cfg.get("decision_tree", {}) if isinstance(models_cfg.get("decision_tree"), dict) else {}
    rf_cfg = models_cfg.get("random_forest", {}) if isinstance(models_cfg.get("random_forest"), dict) else {}

    lr = LogisticRegression(
        max_iter=int(lr_cfg.get("max_iter", 2000)),
        solver=str(lr_cfg.get("solver", "lbfgs")),
        random_state=random_state,
    )
    dt = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=dt_cfg.get("max_depth", None),
        min_samples_split=int(dt_cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(dt_cfg.get("min_samples_leaf", 1)),
    )
    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 400)),
        random_state=random_state,
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
    )

    models = {"logistic_regression": lr, "decision_tree": dt, "random_forest": rf}
    for m in models.values():
        m.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, models


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - train models")
    parser.add_argument("--x", type=str, default=None, help="Feature CSV path (optional override)")
    parser.add_argument("--y", type=str, default=None, help="Label CSV path (optional override)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    try:
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
        paths = get_paths(config, Path(__file__).resolve().parent)
        ensure_output_dirs(paths)
        logger = setup_logger(SCRIPT_PREFIX, paths.logs_dir)

        timer = Timer()
        X_path, y_path = resolve_feature_paths(args.x, args.y)
        logger.info(f"Loading features: {X_path}")
        logger.info(f"Loading labels: {y_path}")

        if X_path.name == "step3__engineered_features.csv":
            df_xy = pd.read_csv(X_path)
            if "ChurnLabel" not in df_xy.columns:
                raise ValueError("step3__engineered_features.csv is missing ChurnLabel.")
            y = df_xy["ChurnLabel"]
            X = df_xy.drop(columns=["ChurnLabel"], errors="ignore")
        else:
            X = pd.read_csv(X_path)
            if y_path.name == "step3__engineered_features.csv":
                df_xy = pd.read_csv(y_path)
                y = df_xy["ChurnLabel"]
            else:
                y = pd.read_csv(y_path)["ChurnLabel"]
        logger.info(f"Loaded X/y. X_shape={X.shape}, y_len={len(y)}, elapsed_s={timer.elapsed_s():.3f}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        msg = f"{type(e).__name__}: {e}"
        print(msg)
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        msg = f"UnexpectedError({type(e).__name__}): {e}"
        print(msg)
        raise SystemExit(1)

    training_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    X_train, X_test, y_train, y_test, models = train_and_split(X, y, training_cfg)
    random_state = int(training_cfg.get("random_state", RANDOM_STATE))
    lr = models["logistic_regression"]
    dt = models["decision_tree"]
    rf = models["random_forest"]

    saved_models = {}
    for name, model in models.items():
        p = paths.models_dir / f"{SCRIPT_PREFIX}__{name}.joblib"
        joblib.dump(model, p)
        saved_models[name] = str(p)

    X_train_path = paths.csv_dir / f"{SCRIPT_PREFIX}__X_train.csv"
    X_test_path = paths.csv_dir / f"{SCRIPT_PREFIX}__X_test.csv"
    y_train_path = paths.csv_dir / f"{SCRIPT_PREFIX}__y_train.csv"
    y_test_path = paths.csv_dir / f"{SCRIPT_PREFIX}__y_test.csv"
    X_train.to_csv(X_train_path, index=False, encoding="utf-8")
    X_test.to_csv(X_test_path, index=False, encoding="utf-8")
    y_train.to_frame("ChurnLabel").to_csv(y_train_path, index=False, encoding="utf-8")
    y_test.to_frame("ChurnLabel").to_csv(y_test_path, index=False, encoding="utf-8")

    summary_path = paths.reports_dir / f"{SCRIPT_PREFIX}__training_summary.txt"
    lines: list[str] = []
    lines.append("Telco Customer Churn - Model Training Summary")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("1) Train/test split")
    lines.append("- Split ratio: 70/30 (train/test)")
    lines.append(f"- Random seed: {random_state}")
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
    lines.append(f"- Models directory: {paths.models_dir}")
    for k, v in saved_models.items():
        lines.append(f"  - {k}: {v}")
    lines.append("- Split datasets (CSV):")
    lines.append(f"  - X_train: {X_train_path}")
    lines.append(f"  - X_test: {X_test_path}")
    lines.append(f"  - y_train: {y_train_path}")
    lines.append(f"  - y_test: {y_test_path}")

    summary_path.write_text("\n".join(lines), encoding="utf-8", errors="replace")

    print("=== Training complete ===")
    print(f"Saved models to: {paths.models_dir}")
    print("Saved split CSVs to outputs/csv/:")
    print(f"- {X_train_path}")
    print(f"- {X_test_path}")
    print(f"- {y_train_path}")
    print(f"- {y_test_path}")
    print(f"Saved training summary: {summary_path}")
    logger.info("Training complete.")
    logger.info(f"Saved models: {saved_models}")
    logger.info(f"Saved splits: X_train={X_train_path}, X_test={X_test_path}, y_train={y_train_path}, y_test={y_test_path}")
    logger.info(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()

