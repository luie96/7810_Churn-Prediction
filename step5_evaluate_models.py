"""
Telco Customer Churn - Module 5/5: Model Evaluation & Result Interpretation

Business goal
- Compare models using consistent metrics and select the best one
- Produce plots (confusion matrix, ROC) and a Top feature list for reporting

Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib shap pyyaml
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from project_utils import Timer, ensure_output_dirs, get_paths, load_config, setup_logger


SCRIPT_PREFIX = "step5_evaluate_models"
UPSTREAM_PREFIX = "step4_train_models"


@dataclass(frozen=True)
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def resolve_config_path(cli_config: str | None) -> Path:
    base = Path(__file__).resolve().parent
    return Path(cli_config).resolve() if cli_config else (base / "config.yaml")


def load_test_split(csv_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    X_test_path = csv_dir / f"{UPSTREAM_PREFIX}__X_test.csv"
    y_test_path = csv_dir / f"{UPSTREAM_PREFIX}__y_test.csv"
    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("Test split CSV not found. Run step4_train_models.py first.")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)["ChurnLabel"].to_numpy()
    return X_test, y_test


def load_models(models_dir: Path) -> dict[str, object]:
    expected = ["logistic_regression", "decision_tree", "random_forest"]
    models: dict[str, object] = {}
    for name in expected:
        p = models_dir / f"{UPSTREAM_PREFIX}__{name}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}. Run step4_train_models.py first.")
        models[name] = joblib.load(p)
    return models


def predict_score(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def evaluate(name: str, model: object, X_test: pd.DataFrame, y_test: np.ndarray) -> ModelResult:
    y_pred = model.predict(X_test)
    y_score = predict_score(model, X_test)
    return ModelResult(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        auc=float(roc_auc_score(y_test, y_score)),
    )


def plot_confusion(name: str, model: object, X_test: pd.DataFrame, y_test: np.ndarray, plots_dir: Path) -> Path:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    out = plots_dir / f"{SCRIPT_PREFIX}__cm_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_roc(name: str, model: object, X_test: pd.DataFrame, y_test: np.ndarray, plots_dir: Path) -> Path:
    y_score = predict_score(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"ROC Curve - {name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    out = plots_dir / f"{SCRIPT_PREFIX}__roc_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def feature_importance(best_model: object, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(best_model, "feature_importances_"):
        imp = np.asarray(best_model.feature_importances_, dtype=float)
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        df["abs_importance"] = df["importance"].abs()
        return df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    if hasattr(best_model, "coef_"):
        coef = np.asarray(best_model.coef_).reshape(-1).astype(float)
        df = pd.DataFrame({"feature": feature_names, "importance": coef})
        df["abs_importance"] = df["importance"].abs()
        return df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    df = pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
    df["abs_importance"] = 0.0
    return df


def plot_shap_summary(
    best_model_name: str,
    best_model: object,
    X_test: pd.DataFrame,
    feature_names: list[str],
    plots_dir: Path,
    config: dict,
    logger,
) -> tuple[Path | None, pd.DataFrame | None]:
    """
    Generate SHAP summary plot for the best model.

    - Tree models: TreeExplainer
    - Linear models: LinearExplainer

    Output path (as required): outputs/plots/step5__shap_summary_{best_model_name}.png
    Returns:
    - plot path (or None if skipped)
    - shap importance dataframe (or None)
    """

    eval_cfg = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    shap_cfg = eval_cfg.get("shap", {}) if isinstance(eval_cfg.get("shap"), dict) else {}
    if not bool(shap_cfg.get("enabled", True)):
        return None, None

    try:
        import shap  # type: ignore
    except ModuleNotFoundError:
        logger.warning("shap is not installed; skipping SHAP analysis.")
        return None, None

    out_path = plots_dir / f"step5__shap_summary_{best_model_name}.png"

    X = X_test.copy()
    if list(X.columns) != feature_names:
        X = X[feature_names]

    # Use a small background set for speed/stability
    max_bg = int(shap_cfg.get("max_background_samples", 200))
    bg = X.sample(n=min(max_bg, len(X)), random_state=0) if len(X) > 0 else X

    try:
        logger.info(f"Running SHAP for best model: {best_model_name}")
        t = Timer()

        if hasattr(best_model, "feature_importances_"):
            explainer = shap.TreeExplainer(best_model, data=bg, feature_perturbation="interventional")
            shap_values = explainer(X)
            values = shap_values.values
        else:
            explainer = shap.LinearExplainer(best_model, bg, feature_perturbation="interventional")
            values = explainer.shap_values(X)

        # Build importance table: mean(|shap|)
        vals = np.asarray(values)
        if vals.ndim == 3:
            # (n_samples, n_classes, n_features) -> take positive class if binary
            vals = vals[:, -1, :]
        mean_abs = np.mean(np.abs(vals), axis=0)
        shap_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )

        # Plot
        plt.figure(figsize=(7.5, 4.8))
        try:
            shap.summary_plot(vals, X, show=False, feature_names=feature_names)
        except Exception:
            # fallback for shap.Explanation
            shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()

        logger.info(f"Saved SHAP summary plot: {out_path} (elapsed_s={t.elapsed_s():.3f})")
        return out_path, shap_imp.reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"SHAP failed: {type(e).__name__}: {e}")
        return None, None


def build_churn_customer_profile_report(
    cleaned_csv_path: Path,
    reports_dir: Path,
    logger,
) -> Path:
    """
    Generate a simple, business-readable high-risk customer profile report with numeric support.
    Output path (as required): outputs/reports/step5__churn_customer_profile.txt
    """

    if not cleaned_csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found for profiling: {cleaned_csv_path}")

    df = pd.read_csv(cleaned_csv_path)
    if "Churn" not in df.columns:
        raise ValueError("Cleaned dataset is missing Churn column.")
    df["ChurnLabel"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    def segment_stats(mask: pd.Series) -> tuple[int, float]:
        n = int(mask.sum())
        rate = float(df.loc[mask, "ChurnLabel"].mean()) if n > 0 else float("nan")
        return n, rate

    segments = []
    if "MonthlyCharges" in df.columns:
        segments.append(("MonthlyCharges > 100", df["MonthlyCharges"] > 100))
    if "tenure" in df.columns:
        segments.append(("Tenure <= 12 months", df["tenure"] <= 12))
    if "Contract" in df.columns:
        segments.append(("Contract = Month-to-month", df["Contract"].astype(str) == "Month-to-month"))
    if "InternetService" in df.columns:
        segments.append(("InternetService = Fiber optic", df["InternetService"].astype(str) == "Fiber optic"))
    if "TechSupport" in df.columns:
        segments.append(("TechSupport = No", df["TechSupport"].astype(str) == "No"))
    if "OnlineSecurity" in df.columns:
        segments.append(("OnlineSecurity = No", df["OnlineSecurity"].astype(str) == "No"))
    if "PaymentMethod" in df.columns:
        segments.append(("PaymentMethod = Electronic check", df["PaymentMethod"].astype(str) == "Electronic check"))

    lines: list[str] = []
    lines.append("Telco Customer Churn - High-Risk Customer Profile")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Overall churn rate: {df['ChurnLabel'].mean():.4f} (n={len(df)})")
    lines.append("")
    lines.append("Segment-based churn signals (sample size and churn rate):")

    for name, mask in segments:
        n, rate = segment_stats(mask)
        lines.append(f"- {name}: n={n}, churn_rate={rate:.4f}")

    out_path = reports_dir / "step5__churn_customer_profile.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8", errors="replace")
    logger.info(f"Saved churn customer profile report: {out_path}")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - evaluate models")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    try:
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
        paths = get_paths(config, Path(__file__).resolve().parent)
        ensure_output_dirs(paths)
        logger = setup_logger(SCRIPT_PREFIX, paths.logs_dir)

        timer = Timer()
        X_test, y_test = load_test_split(paths.csv_dir)
        models = load_models(paths.models_dir)
        logger.info(f"Loaded test split and models. X_test_shape={X_test.shape}, elapsed_s={timer.elapsed_s():.3f}")
    except (FileNotFoundError, ValueError) as e:
        msg = f"{type(e).__name__}: {e}"
        print(msg)
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        msg = f"UnexpectedError({type(e).__name__}): {e}"
        print(msg)
        raise SystemExit(1)

    results = [evaluate(name, m, X_test, y_test) for name, m in models.items()]
    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("auc", ascending=False).reset_index(drop=True)

    plot_paths: list[str] = []
    for name, m in models.items():
        plot_paths.append(str(plot_confusion(name, m, X_test, y_test, paths.plots_dir)))
        plot_paths.append(str(plot_roc(name, m, X_test, y_test, paths.plots_dir)))

    best_row = metrics_df.sort_values(["auc", "f1"], ascending=False).iloc[0]
    best_name = str(best_row["name"])
    best_model = models[best_name]

    fi_df = feature_importance(best_model, list(X_test.columns))
    eval_cfg = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    top_k = int(eval_cfg.get("importance_top_k", 15))
    fi_top = fi_df.head(top_k)

    # SHAP analysis (optional, config-controlled)
    shap_plot_path, shap_imp_df = plot_shap_summary(
        best_model_name=best_name,
        best_model=best_model,
        X_test=X_test,
        feature_names=list(X_test.columns),
        plots_dir=paths.plots_dir,
        config=config,
        logger=logger,
    )

    # High-risk customer profile report (based on cleaned dataset)
    cleaned_csv = paths.csv_dir / "step2_preprocess__telco_cleaned.csv"
    profile_path = None
    try:
        profile_path = build_churn_customer_profile_report(cleaned_csv, paths.reports_dir, logger)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to build customer profile report: {type(e).__name__}: {e}")

    metrics_path = paths.csv_dir / f"{SCRIPT_PREFIX}__model_metrics.csv"
    fi_path = paths.csv_dir / f"{SCRIPT_PREFIX}__best_model_feature_importance.csv"
    report_path = paths.reports_dir / f"{SCRIPT_PREFIX}__evaluation_report.txt"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    fi_df.to_csv(fi_path, index=False, encoding="utf-8")

    lines: list[str] = []
    lines.append("Telco Customer Churn - Model Evaluation Report")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("1) Metrics comparison (Accuracy / Precision / Recall / F1 / AUC)")
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("2) Best model selection")
    lines.append(f"- Best model (primary: AUC, secondary: F1): {best_name}")
    lines.append(
        f"- Best metrics: Accuracy={best_row['accuracy']:.4f}, Precision={best_row['precision']:.4f}, "
        f"Recall={best_row['recall']:.4f}, F1={best_row['f1']:.4f}, AUC={best_row['auc']:.4f}"
    )
    lines.append("")
    lines.append("3) Plots")
    lines.append(f"- Plots directory: {paths.plots_dir}")
    for p in plot_paths:
        lines.append(f"  - {p}")
    lines.append("")
    lines.append("4) Top drivers (feature importance) and interpretation hints")
    lines.append(fi_top.to_string(index=False))
    lines.append("")
    lines.append("5) SHAP explainability (best model)")
    if shap_plot_path and shap_imp_df is not None:
        lines.append(f"- SHAP summary plot: {shap_plot_path}")
        lines.append("- Top SHAP features (mean |SHAP|):")
        lines.append(shap_imp_df.head(top_k).to_string(index=False))
    else:
        lines.append("- SHAP analysis skipped or failed (check logs and requirements).")
    lines.append("")
    lines.append("6) High-risk customer profile (business segmentation)")
    if profile_path:
        lines.append(f"- Profile report saved: {profile_path}")
    else:
        lines.append("- Profile report not generated (check logs).")
    lines.append("")
    lines.append("Interpretation hints (for report writing):")
    lines.append("- Contract type, tenure, and monthly charges often dominate churn risk in telco datasets.")
    lines.append("- Service add-ons like TechSupport / OnlineSecurity can indicate stickiness; consider targeted bundles.")
    lines.append("- Payment method importance may reflect friction or pricing sensitivity; improve autopay adoption and billing clarity.")

    report_path.write_text("\n".join(lines), encoding="utf-8", errors="replace")

    print("=== Evaluation complete ===")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Best-model feature importance CSV: {fi_path}")
    print(f"Evaluation report: {report_path}")
    print(f"Plots saved to: {paths.plots_dir}")
    logger.info(f"Saved metrics: {metrics_path}")
    logger.info(f"Saved feature importance: {fi_path}")
    logger.info(f"Saved evaluation report: {report_path}")
    logger.info(f"Saved plots to: {paths.plots_dir}")


if __name__ == "__main__":
    main()

