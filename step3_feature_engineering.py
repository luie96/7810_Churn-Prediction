"""
Telco Customer Churn - Module 3/5: Feature Engineering & Feature Selection

Business goal
- Convert business fields into a model-ready numeric feature matrix
- Produce correlation-based insights to support retention strategy narratives

Dependencies
pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_PREFIX = "step3_feature_engineering"
UPSTREAM_PREFIX = "step2_preprocess"


def ensure_dirs() -> dict[str, Path]:
    base = Path(__file__).resolve().parent
    outputs = base / "outputs"
    csv_dir = outputs / "csv"
    reports_dir = outputs / "reports"

    outputs.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base, "outputs": outputs, "csv": csv_dir, "reports": reports_dir}


def resolve_cleaned_input_path(cli_path: str | None) -> Path:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "outputs" / "csv" / f"{UPSTREAM_PREFIX}__telco_cleaned.csv",
            here / "outputs" / f"{UPSTREAM_PREFIX}__telco_cleaned.csv",
        ]
    )

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        "Cleaned dataset not found. Please run step2_preprocess.py first, or pass --input explicitly."
    )


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    numeric_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]

    binary_cols = [
        c
        for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]
        if c in df.columns
    ]

    multi_cols = [
        c
        for c in [
            "gender",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ]
        if c in df.columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), binary_cols + multi_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, binary_cols, multi_cols


def feature_churn_correlations(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    y = y.astype(float).reshape(-1)
    rows = []
    for i, name in enumerate(feature_names):
        xi = X[:, i].astype(float)
        if np.nanstd(xi) == 0 or np.nanstd(y) == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(xi, y)[0, 1])
            if np.isnan(r):
                r = 0.0
        rows.append((name, r, abs(r)))

    df_corr = pd.DataFrame(rows, columns=["feature", "corr_with_churn", "abs_corr"])
    return df_corr.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - feature engineering")
    parser.add_argument("--input", type=str, default=None, help="Cleaned CSV path (default: outputs/csv/step2_preprocess__telco_cleaned.csv)")
    args = parser.parse_args()

    dirs = ensure_dirs()
    input_path = resolve_cleaned_input_path(args.input)
    df = pd.read_csv(input_path)

    # Target variable
    if "ChurnLabel" in df.columns:
        y = df["ChurnLabel"].to_numpy()
    elif "Churn" in df.columns:
        y = df["Churn"].map({"Yes": 1, "No": 0}).to_numpy()
    else:
        raise ValueError("Missing target column: Churn or ChurnLabel.")

    drop_cols = [c for c in ["customerID", "ChurnLabel", "Churn"] if c in df.columns]
    X_raw = df.drop(columns=drop_cols, errors="ignore").copy()

    preprocessor, numeric_cols, binary_cols, multi_cols = build_preprocessor(X_raw)
    X = preprocessor.fit_transform(X_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    corr_df = feature_churn_correlations(X, y, feature_names)

    top_n_for_matrix = 30
    top_features = corr_df.head(top_n_for_matrix)["feature"].tolist()
    top_idx = [feature_names.index(f) for f in top_features]
    X_top = X[:, top_idx]
    corr_matrix = pd.DataFrame(np.corrcoef(X_top, rowvar=False), index=top_features, columns=top_features)

    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame({"ChurnLabel": y.astype(int)})

    X_path = dirs["csv"] / f"{SCRIPT_PREFIX}__features_X.csv"
    y_path = dirs["csv"] / f"{SCRIPT_PREFIX}__labels_y.csv"
    corr_path = dirs["csv"] / f"{SCRIPT_PREFIX}__feature_churn_correlation.csv"
    matrix_path = dirs["csv"] / f"{SCRIPT_PREFIX}__top_feature_correlation_matrix.csv"
    meta_path = dirs["reports"] / f"{SCRIPT_PREFIX}__metadata.json"
    summary_path = dirs["reports"] / f"{SCRIPT_PREFIX}__summary.txt"

    X_df.to_csv(X_path, index=False, encoding="utf-8")
    y_df.to_csv(y_path, index=False, encoding="utf-8")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8")
    corr_matrix.to_csv(matrix_path, encoding="utf-8")

    meta = {
        "input_cleaned_csv": str(input_path),
        "numeric_cols_scaled": numeric_cols,
        "categorical_cols_onehot": binary_cols + multi_cols,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")

    top_k = 15
    top_k_df = corr_df.head(top_k)

    summary_lines: list[str] = []
    summary_lines.append("Telco Customer Churn - Feature Engineering Summary")
    summary_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Input dataset: {input_path}")
    summary_lines.append("")
    summary_lines.append("1) Transformation strategy")
    summary_lines.append(f"- Numeric scaled (StandardScaler): {', '.join(numeric_cols) if numeric_cols else 'None'}")
    summary_lines.append(
        "- Categorical encoded (One-Hot): "
        + (", ".join(binary_cols + multi_cols) if (binary_cols or multi_cols) else "None")
    )
    summary_lines.append("")
    summary_lines.append("2) Top features by absolute correlation with churn (heuristic)")
    summary_lines.append(top_k_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Note: correlation is for interpretability and triage, not causality.")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8", errors="replace")

    print("=== Feature engineering complete ===")
    print(f"X features: {X_path}")
    print(f"y labels: {y_path}")
    print(f"Feature-churn correlation: {corr_path}")
    print(f"Top-feature correlation matrix: {matrix_path}")
    print(f"Metadata: {meta_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

