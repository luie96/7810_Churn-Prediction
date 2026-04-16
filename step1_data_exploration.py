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
import seaborn as sns
from matplotlib import pyplot as plt

from project_utils import Timer, ensure_output_dirs, get_paths, load_config, setup_logger


SCRIPT_PREFIX = "step1_data_exploration"


def resolve_config_path(cli_config: str | None) -> Path:
    base = Path(__file__).resolve().parent
    return Path(cli_config).resolve() if cli_config else (base / "config.yaml")


def resolve_dataset_path(inputs_dir: Path, config: dict, cli_path: str | None) -> Path:
    """
    Project convention: raw data is loaded from the inputs/ folder.
    - If --csv is provided, use it as an explicit override.
    - Otherwise, look for common filenames in inputs/.
    """

    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    raw_candidates = (config.get("data") or {}).get("raw_candidates", []) if isinstance(config.get("data"), dict) else []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    if not raw_candidates:
        raw_candidates = ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
    candidates.extend([inputs_dir / str(name) for name in raw_candidates])

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


def save_plots_png_only(
    plots_dir: Path,
    report_prefix: str,
    df: pd.DataFrame,
    churn_dist: pd.DataFrame,
    cat_dists: dict[str, pd.DataFrame],
) -> list[Path]:
    """
    Save EDA plots as PNG only under outputs/plots/.
    """

    out_paths: list[Path] = []

    def save_fig(fig, name: str) -> None:
        png = plots_dir / f"{report_prefix}__{name}.png"
        fig.savefig(png, dpi=180)
        out_paths.append(png)

    # Churn bar chart
    fig = plt.figure(figsize=(5.6, 4.2))
    ax = fig.add_subplot(111)
    churn_dist["count"].plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("Churn distribution (count)")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    fig.tight_layout()
    save_fig(fig, "churn_bar")
    plt.close(fig)

    # Numeric histograms
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in df.columns:
            continue
        series = df[col]
        if col == "TotalCharges":
            series = pd.to_numeric(series.astype(str).str.strip().replace({"": np.nan}), errors="coerce")
        fig = plt.figure(figsize=(6.2, 4.2))
        ax = fig.add_subplot(111)
        sns.histplot(series, bins=30, kde=True, ax=ax)
        ax.set_title(f"{col} distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        save_fig(fig, f"{col}_hist")
        plt.close(fig)

    # Categorical bar charts
    for col, dist in cat_dists.items():
        fig = plt.figure(figsize=(6.6, 4.2))
        ax = fig.add_subplot(111)
        dist["count"].plot(kind="bar", ax=ax, color="#54A24B")
        ax.set_title(f"{col} distribution (count)")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        save_fig(fig, f"{col}_bar")
        plt.close(fig)

    return out_paths

def validate_required_columns(df: pd.DataFrame, config: dict) -> None:
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    required = data_cfg.get("required_columns", []) if isinstance(data_cfg, dict) else []
    if not required:
        return
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


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
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    try:
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
        paths = get_paths(config, Path(__file__).resolve().parent)
        ensure_output_dirs(paths)
        logger = setup_logger(SCRIPT_PREFIX, paths.logs_dir)

        timer = Timer()
        csv_path = resolve_dataset_path(paths.inputs_dir, config, args.csv)
        logger.info(f"Loading dataset: {csv_path}")
        df = load_raw_csv(csv_path)
        validate_required_columns(df, config)
        logger.info(f"Dataset loaded. rows={len(df)}, cols={df.shape[1]}, elapsed_s={timer.elapsed_s():.3f}")
    except (FileNotFoundError, ValueError) as e:
        # Friendly errors for common issues
        msg = f"{type(e).__name__}: {e}"
        print(msg)
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        msg = f"UnexpectedError({type(e).__name__}): {e}"
        print(msg)
        raise SystemExit(1)

    fd = field_dictionary()
    n_rows, n_cols = df.shape
    churn_dist = churn_distribution(df)

    cat_cols = [c for c in ["gender", "Contract", "InternetService", "PaymentMethod", "SeniorCitizen"] if c in df.columns]
    cat_dists = categorical_distribution(df, cat_cols)

    dq = data_quality_checks(df)
    ns = numeric_stats(df)

    # Plots (PNG only)
    plot_paths = save_plots_png_only(
        plots_dir=paths.plots_dir,
        report_prefix="step1",
        df=df,
        churn_dist=churn_dist,
        cat_dists=cat_dists,
    )
    logger.info(f"Saved step1 plots: {len(plot_paths)} PNG files")

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

    report_path = save_report(paths.reports_dir, "\n".join(lines))
    print(f"Saved report: {report_path}")
    logger.info(f"Saved report: {report_path}")
    print(f"Saved EDA plots (PNG) under: {paths.plots_dir}")


if __name__ == "__main__":
    main()

