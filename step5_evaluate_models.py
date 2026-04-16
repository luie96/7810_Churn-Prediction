"""
Telco Customer Churn - Module 5/5: Model Evaluation & Result Interpretation

Business goal
- Compare models using consistent metrics and select the best one
- Produce plots (confusion matrix, ROC) and a Top feature list for reporting

Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib
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


def ensure_dirs() -> dict[str, Path]:
    base = Path(__file__).resolve().parent
    outputs = base / "outputs"
    csv_dir = outputs / "csv"
    reports_dir = outputs / "reports"
    plots_dir = outputs / "plots"
    models_dir = outputs / "models"

    outputs.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base, "outputs": outputs, "csv": csv_dir, "reports": reports_dir, "plots": plots_dir, "models": models_dir}


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


def main() -> None:
    dirs = ensure_dirs()
    X_test, y_test = load_test_split(dirs["csv"])
    models = load_models(dirs["models"])

    results = [evaluate(name, m, X_test, y_test) for name, m in models.items()]
    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("auc", ascending=False).reset_index(drop=True)

    plot_paths: list[str] = []
    for name, m in models.items():
        plot_paths.append(str(plot_confusion(name, m, X_test, y_test, dirs["plots"])))
        plot_paths.append(str(plot_roc(name, m, X_test, y_test, dirs["plots"])))

    best_row = metrics_df.sort_values(["auc", "f1"], ascending=False).iloc[0]
    best_name = str(best_row["name"])
    best_model = models[best_name]

    fi_df = feature_importance(best_model, list(X_test.columns))
    fi_top = fi_df.head(15)

    metrics_path = dirs["csv"] / f"{SCRIPT_PREFIX}__model_metrics.csv"
    fi_path = dirs["csv"] / f"{SCRIPT_PREFIX}__best_model_feature_importance.csv"
    report_path = dirs["reports"] / f"{SCRIPT_PREFIX}__evaluation_report.txt"

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
    lines.append(f"- Plots directory: {dirs['plots']}")
    for p in plot_paths:
        lines.append(f"  - {p}")
    lines.append("")
    lines.append("4) Top drivers (feature importance) and interpretation hints")
    lines.append(fi_top.to_string(index=False))
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
    print(f"Plots saved to: {dirs['plots']}")


if __name__ == "__main__":
    main()

