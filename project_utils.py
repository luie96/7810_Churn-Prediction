from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    inputs_dir: Path
    outputs_dir: Path
    csv_dir: Path
    reports_dir: Path
    plots_dir: Path
    models_dir: Path
    logs_dir: Path


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Invalid config.yaml format: expected a YAML mapping at the root.")
    return data


def get_paths(config: dict[str, Any], project_root: Path) -> ProjectPaths:
    paths_cfg = (config.get("paths") or {}) if isinstance(config.get("paths"), dict) else {}

    def p(key: str, default: str) -> Path:
        val = paths_cfg.get(key, default)
        return (project_root / str(val)).resolve()

    outputs_dir = p("outputs_dir", "outputs")
    return ProjectPaths(
        project_root=project_root.resolve(),
        inputs_dir=p("inputs_dir", "inputs"),
        outputs_dir=outputs_dir,
        csv_dir=p("csv_dir", "outputs/csv"),
        reports_dir=p("reports_dir", "outputs/reports"),
        plots_dir=p("plots_dir", "outputs/plots"),
        models_dir=p("models_dir", "outputs/models"),
        logs_dir=p("logs_dir", "outputs/logs"),
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.csv_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def setup_logger(script_name: str, logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")
    log_path = logs_dir / f"{script_name}_{date_tag}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    logger.debug(f"Logger initialized. Log file: {log_path}")
    return logger


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

