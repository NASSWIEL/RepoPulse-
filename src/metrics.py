"""Forecast quality metrics and held-out backtest."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.forecast import Forecaster


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.where((y_true == 0) & (y_pred == 0), 0.0,
                     num / np.where(den == 0, 1.0, den))
    return float(100.0 * np.mean(ratio))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def backtest(
    series: pd.Series,
    holdout: int,
    horizon: int,
    forecasters: Sequence[Forecaster],
) -> pd.DataFrame:
    """Hold out the last `holdout` months and forecast the first `horizon` of them."""
    if holdout < horizon:
        raise ValueError(f"holdout ({holdout}) must be >= horizon ({horizon})")
    if len(series) < holdout + 6:
        raise ValueError(f"series too short for backtest: {len(series)} < {holdout + 6}")

    train = series.iloc[:-holdout]
    truth = series.iloc[-holdout : -holdout + horizon].values.astype(float)

    rows = []
    for f in forecasters:
        res = f.forecast(train, horizon=horizon)
        rows.append({
            "model": res.model_name,
            "smape": smape(truth, res.mean),
            "mae": mae(truth, res.mean),
            "latency_ms": res.latency_ms,
        })
    return pd.DataFrame(rows)
