"""
Single entrypoint: run the full Telco churn pipeline (5 steps).

Examples
1) Recommended: put the raw CSV under inputs/WA_Fn-UseC_-Telco-Customer-Churn.csv, then run:
   python main.py

2) If your raw CSV is elsewhere, pass --csv. The file will be copied into inputs/ with the canonical filename:
   python main.py --csv "D:\\path\\to\\WA_Fn-UseC_-Telco-Customer-Churn.csv"
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


RAW_CANONICAL_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_CONFIG_NAME = "config.yaml"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_inputs_dir() -> Path:
    p = project_root() / "inputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_outputs_dir() -> Path:
    p = project_root() / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_raw_csv_to_inputs(csv_path: str | None) -> Path:
    """
    If --csv is provided, copy it into inputs/ as the canonical input file.
    Otherwise, require the canonical (or fallback) filename to already exist under inputs/.
    """

    inputs_dir = ensure_inputs_dir()

    if csv_path:
        src = Path(csv_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"--csv points to a missing file: {src}")

        dst = inputs_dir / RAW_CANONICAL_NAME
        shutil.copy2(src, dst)
        return dst

    # 默认情况：从 inputs/ 读取
    candidates = [
        inputs_dir / RAW_CANONICAL_NAME,
        inputs_dir / "Telco-Customer-Churn.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        "Raw CSV not found under inputs/.\n"
        f"- Please put the dataset under: {inputs_dir}\n"
        f"- Recommended filename: {RAW_CANONICAL_NAME}\n"
        "- Or run: python main.py --csv \"path\\to\\csv\""
    )


def run_step(script_name: str) -> None:
    script_path = project_root() / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n>>> Running: {script_name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def run_step_with_config(script_name: str, config_path: Path) -> None:
    script_path = project_root() / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    print(f"\n>>> Running: {script_name}")
    subprocess.run([sys.executable, str(script_path), "--config", str(config_path)], check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - main pipeline runner")
    parser.add_argument("--csv", type=str, default=None, help="Raw Telco churn CSV path (will be copied into inputs/)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    ensure_outputs_dir()
    staged = stage_raw_csv_to_inputs(args.csv)
    print(f"Raw dataset ready: {staged}")

    config_path = Path(args.config).resolve() if args.config else (project_root() / DEFAULT_CONFIG_NAME)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Fixed order (aligned to the 5-step pipeline)
    run_step_with_config("step1_data_exploration.py", config_path)
    run_step_with_config("step2_preprocess.py", config_path)
    run_step_with_config("step3_feature_engineering.py", config_path)
    run_step_with_config("step4_train_models.py", config_path)
    run_step_with_config("step5_evaluate_models.py", config_path)

    print("\nPipeline finished. Outputs directory:")
    print(f"- {project_root() / 'outputs'}")


if __name__ == "__main__":
    main()

