"""
Telco Customer Churn - Module 1/5: Data Exploration & Dataset Review

Business goal
- Understand customer composition and churn distribution
- Identify potential churn drivers (contract / services / payment patterns)
- Check data quality risks before modeling (missing / duplicates / type issues)

Dependencies
pip install pandas numpy
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SCRIPT_PREFIX = "step1_data_exploration"


def ensure_dirs() -> dict[str, Path]:
    base = Path(__file__).resolve().parent
    inputs = base / "inputs"
    outputs = base / "outputs"
    csv_dir = outputs / "csv"
    reports_dir = outputs / "reports"

    outputs.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base, "inputs": inputs, "outputs": outputs, "csv": csv_dir, "reports": reports_dir}


def resolve_dataset_path(inputs_dir: Path, cli_path: str | None) -> Path:
    """
    Project convention: raw data is loaded from the inputs/ folder.
    - If --csv is provided, use it as an explicit override.
    - Otherwise, look for common filenames in inputs/.
    """

    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    candidates.extend(
        [
            inputs_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
            inputs_dir / "Telco-Customer-Churn.csv",
        ]
    )

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        "Raw CSV not found.\n"
        f"- Please put the dataset under: {inputs_dir}\n"
        "- Recommended filename: WA_Fn-UseC_-Telco-Customer-Churn.csv\n"
        "- Or pass an explicit path via --csv"
    )


def field_dictionary() -> Dict[str, Tuple[str, str]]:
    """
    Field definitions aligned to the typical Telco churn dataset.
    Returns: column -> (category, business meaning)
    """

    return {
        "customerID": ("Customer identifier", "Unique customer ID"),
        "gender": ("Demographics", "Gender"),
        "SeniorCitizen": ("Demographics", "Senior citizen (0/1)"),
        "Partner": ("Demographics", "Has a partner (Yes/No)"),
        "Dependents": ("Demographics", "Has dependents (Yes/No)"),
        "tenure": ("Account", "Tenure in months"),
        "PhoneService": ("Services", "Has phone service (Yes/No)"),
        "MultipleLines": ("Services", "Multiple lines (Yes/No/No phone service)"),
        "InternetService": ("Services", "Internet service type (DSL/Fiber optic/No)"),
        "OnlineSecurity": ("Services", "Online security (Yes/No/No internet service)"),
        "OnlineBackup": ("Services", "Online backup (Yes/No/No internet service)"),
        "DeviceProtection": ("Services", "Device protection (Yes/No/No internet service)"),
        "TechSupport": ("Services", "Tech support (Yes/No/No internet service)"),
        "StreamingTV": ("Services", "Streaming TV (Yes/No/No internet service)"),
        "StreamingMovies": ("Services", "Streaming movies (Yes/No/No internet service)"),
        "Contract": ("Account", "Contract type (Month-to-month/One year/Two year)"),
        "PaperlessBilling": ("Account", "Paperless billing (Yes/No)"),
        "PaymentMethod": ("Account", "Payment method"),
        "MonthlyCharges": ("Account", "Monthly charges"),
        "TotalCharges": ("Account", "Total charges (often stored as string; may contain blanks)"),
        "Churn": ("Target", "Churn label (Yes/No)"),
    }


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    # Keep raw types; TotalCharges commonly contains blanks/spaces.
    return pd.read_csv(csv_path)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().replace({"": np.nan}), errors="coerce")


def churn_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Churn"].value_counts(dropna=False)
    ratios = (counts / len(df)).rename("ratio")
    return pd.concat([counts.rename("count"), ratios], axis=1)


def categorical_distribution(df: pd.DataFrame, cols: list[str]) -> Dict[str, pd.DataFrame]:
    dist: Dict[str, pd.DataFrame] = {}
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        dist[c] = pd.concat([vc.rename("count"), (vc / len(df)).rename("ratio")], axis=1)
    return dist


def data_quality_checks(df: pd.DataFrame) -> Dict[str, object]:
    result: Dict[str, object] = {}

    missing = df.isna().sum().sort_values(ascending=False)
    result["missing_values"] = missing[missing > 0].to_dict()

    if "TotalCharges" in df.columns:
        tc_raw = df["TotalCharges"].astype(str)
        result["TotalCharges_blank_string_count"] = int((tc_raw.str.strip() == "").sum())

    result["duplicate_rows"] = int(df.duplicated().sum())
    if "customerID" in df.columns:
        result["duplicate_customerID"] = int(df["customerID"].duplicated().sum())

    result["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}

    numeric_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    numeric_df = df.copy()
    if "TotalCharges" in numeric_cols:
        numeric_df["TotalCharges_num"] = safe_to_numeric(numeric_df["TotalCharges"])
        numeric_cols_eff = ["tenure", "MonthlyCharges", "TotalCharges_num"]
    else:
        numeric_cols_eff = numeric_cols

    outlier_hints: Dict[str, Dict[str, float]] = {}
    for c in numeric_cols_eff:
        s = pd.to_numeric(numeric_df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        q1, q3 = np.nanpercentile(s, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_hints[c] = {
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "iqr_lower_bound": float(lower),
            "iqr_upper_bound": float(upper),
            "outlier_count_iqr_rule": int(((s < lower) | (s > upper)).sum(skipna=True)),
        }
    result["numeric_outlier_hints"] = outlier_hints
    return result


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    if "TotalCharges" in tmp.columns:
        tmp["TotalCharges"] = safe_to_numeric(tmp["TotalCharges"])
    cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in tmp.columns]
    desc = tmp[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc.rename(columns={"50%": "median"}, inplace=True)
    return desc


def save_report(reports_dir: Path, content: str) -> Path:
    p = reports_dir / f"{SCRIPT_PREFIX}__data_exploration_summary.txt"
    p.write_text(content, encoding="utf-8", errors="replace")
    return p


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - data exploration")
    parser.add_argument("--csv", type=str, default=None, help="Raw Telco churn CSV path (optional override)")
    args = parser.parse_args()

    dirs = ensure_dirs()
    csv_path = resolve_dataset_path(dirs["inputs"], args.csv)
    df = load_raw_csv(csv_path)

    fd = field_dictionary()
    n_rows, n_cols = df.shape
    churn_dist = churn_distribution(df)

    cat_cols = [c for c in ["gender", "Contract", "InternetService", "PaymentMethod", "SeniorCitizen"] if c in df.columns]
    cat_dists = categorical_distribution(df, cat_cols)

    dq = data_quality_checks(df)
    ns = numeric_stats(df)

    print("=== Dataset overview ===")
    print(f"Dataset path: {csv_path}")
    print(f"Rows: {n_rows}")
    print(f"Columns: {n_cols}")
    print()

    print("=== Field definitions (for reporting) ===")
    for col in df.columns:
        cat, meaning = fd.get(col, ("Uncategorized", "(please define)"))
        print(f"- {col}: [{cat}] {meaning}")
    print()

    print("=== Churn distribution ===")
    print(churn_dist.to_string())
    print()

    print("=== Key categorical distributions (count/ratio) ===")
    for c, t in cat_dists.items():
        print(f"\n[{c}]")
        print(t.to_string())
    print()

    print("=== Data quality summary ===")
    print(json.dumps(dq, ensure_ascii=False, indent=2))
    print()

    print("=== Numeric descriptive statistics ===")
    print(ns.to_string())
    print()

    lines: list[str] = []
    lines.append("Telco Customer Churn - Data Exploration Summary")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset path: {csv_path}")
    lines.append("")
    lines.append("1) Dataset size")
    lines.append(f"- Rows: {n_rows}")
    lines.append(f"- Columns: {n_cols}")
    lines.append("")
    lines.append("2) Churn distribution")
    lines.append(churn_dist.to_string())
    lines.append("")
    lines.append("3) Key categorical distributions")
    for c, t in cat_dists.items():
        lines.append(f"[{c}]")
        lines.append(t.to_string())
        lines.append("")
    lines.append("4) Data quality checks")
    lines.append(json.dumps(dq, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("5) Numeric descriptive statistics")
    lines.append(ns.to_string())
    lines.append("")

    report_path = save_report(dirs["reports"], "\n".join(lines))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

