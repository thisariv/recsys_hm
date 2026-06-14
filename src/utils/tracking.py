"""Настройка локального MLflow-трекинга."""
from __future__ import annotations

from typing import Any

import mlflow


def init_mlflow(config: dict[str, Any]) -> None:
    """Установить tracking_uri и experiment из секции `mlflow` конфига."""
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlruns")
    experiment = mlflow_cfg.get("experiment", "default")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
