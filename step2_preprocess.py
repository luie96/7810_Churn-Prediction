"""
Telco Customer Churn - Module 2/5: Data Cleaning & Preprocessing

Business goal
- Produce a clean, standardized dataset for modeling
- Handle the known pain point: TotalCharges missing/blank strings + numeric conversion

Dependencies
pip install pandas numpy
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PREFIX = "step2_preprocess"


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


def safe_to_numeric_totalcharges(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan}), errors="coerce")


def clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps: list[str] = []
    df = df_raw.copy()

    steps.append(f"Raw dataset size: rows={len(df)}, cols={df.shape[1]}")

    # 1) Deduplication: customerID should be unique.
    if "customerID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["customerID"], keep="first")
        steps.append(f"Deduplicate by customerID: {before} -> {len(df)} (dropped {before - len(df)})")
    else:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        steps.append(f"Deduplicate by full row: {before} -> {len(df)} (dropped {before - len(df)})")

    # 2) TotalCharges: convert to numeric and handle missing.
    if "TotalCharges" in df.columns:
        blank_count = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
        df["TotalCharges"] = safe_to_numeric_totalcharges(df["TotalCharges"])
        missing_after = int(df["TotalCharges"].isna().sum())
        steps.append(
            "TotalCharges cleaning: treat blank strings as missing and convert to numeric. "
            f"blank_strings={blank_count}, missing_after_conversion={missing_after}."
        )

        # Business logic: tenure=0 implies newly joined; TotalCharges should be 0.
        if "tenure" in df.columns:
            m = (df["tenure"] == 0) & (df["TotalCharges"].isna())
            filled = int(m.sum())
            df.loc[m, "TotalCharges"] = 0.0
            steps.append(f"Fill TotalCharges=0 when tenure=0 and TotalCharges missing: filled {filled} rows.")

        # Remaining missing: fill with median (robust; keeps linear models stable).
        remaining = int(df["TotalCharges"].isna().sum())
        if remaining > 0:
            med = float(df["TotalCharges"].median(skipna=True))
            df["TotalCharges"] = df["TotalCharges"].fillna(med)
            steps.append(f"Fill remaining TotalCharges missing with median: missing={remaining}, median={med:.4f}.")

    # 3) SeniorCitizen: normalize 0/1 -> Yes/No (clearer for one-hot later).
    if "SeniorCitizen" in df.columns:
        before_unique = df["SeniorCitizen"].unique().tolist()
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"].astype(str))
        after_unique = df["SeniorCitizen"].unique().tolist()
        steps.append(f"Normalize SeniorCitizen: before={before_unique}, after={after_unique}.")

    # 4) Churn label: add numeric label for modeling.
    if "Churn" in df.columns:
        df["ChurnLabel"] = df["Churn"].map({"Yes": 1, "No": 0})
        unmapped = int(df["ChurnLabel"].isna().sum())
        if unmapped > 0:
            df["ChurnLabel"] = pd.to_numeric(df["ChurnLabel"], errors="coerce")
        steps.append(f"Encode churn label: add ChurnLabel (Yes=1, No=0). Unmapped={unmapped}.")

    # 5) Remove obviously invalid rows (negative values).
    invalid = pd.Series(False, index=df.index)
    if "tenure" in df.columns:
        invalid |= pd.to_numeric(df["tenure"], errors="coerce") < 0
    if "MonthlyCharges" in df.columns:
        invalid |= pd.to_numeric(df["MonthlyCharges"], errors="coerce") < 0
    if "TotalCharges" in df.columns:
        invalid |= pd.to_numeric(df["TotalCharges"], errors="coerce") < 0

    invalid_count = int(invalid.sum())
    if invalid_count > 0:
        before = len(df)
        df = df.loc[~invalid].copy()
        steps.append(f"Drop invalid rows (negative charges/tenure): {before} -> {len(df)} (dropped {invalid_count}).")
    else:
        steps.append("No invalid rows found (negative charges/tenure).")

    # 6) Final dtype normalization for numeric columns.
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    steps.append(f"Cleaned dataset size: rows={len(df)}, cols={df.shape[1]}")
    return df, steps


def save_artifacts(csv_dir: Path, reports_dir: Path, df_clean: pd.DataFrame, steps: list[str]) -> tuple[Path, Path]:
    data_path = csv_dir / f"{SCRIPT_PREFIX}__telco_cleaned.csv"
    steps_path = reports_dir / f"{SCRIPT_PREFIX}__preprocess_steps_summary.txt"

    df_clean.to_csv(data_path, index=False, encoding="utf-8")

    content = "\n".join(
        [
            "Telco Customer Churn - Preprocessing Steps Summary",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            *[f"- {s}" for s in steps],
            "",
            "Note: this file documents key preprocessing decisions and dataset size changes for reporting.",
        ]
    )
    steps_path.write_text(content, encoding="utf-8", errors="replace")
    return data_path, steps_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - preprocessing")
    parser.add_argument("--csv", type=str, default=None, help="Raw Telco churn CSV path (optional override)")
    args = parser.parse_args()

    dirs = ensure_dirs()
    csv_path = resolve_dataset_path(dirs["inputs"], args.csv)
    df_raw = pd.read_csv(csv_path)

    df_clean, steps = clean_dataset(df_raw)
    cleaned_path, steps_path = save_artifacts(dirs["csv"], dirs["reports"], df_clean, steps)

    print("=== Preprocessing complete ===")
    print(f"Saved cleaned dataset: {cleaned_path}")
    print(f"Saved steps summary: {steps_path}")
    print()
    print("Steps (copy/paste into report):")
    for s in steps:
        print("-", s)


if __name__ == "__main__":
    main()

