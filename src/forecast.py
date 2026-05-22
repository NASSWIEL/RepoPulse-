"""Forecasting models: naive seasonal baseline, Chronos zero-shot, Chronos fine-tuned."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    latency_ms: float
    model_name: str


class Forecaster(Protocol):
    def forecast(self, series: pd.Series, horizon: int) -> ForecastResult: ...


class NaiveSeasonal:
    """Repeat the last `period` observations. Falls back to last value if series shorter."""
    model_name = "naive_seasonal"

    def __init__(self, period: int = 12):
        self.period = period

    def forecast(self, series: pd.Series, horizon: int) -> ForecastResult:
        t0 = time.perf_counter()
        values = series.values.astype(float)
        if len(values) < self.period:
            last = values[-1] if len(values) else 0.0
            mean = np.full(horizon, last)
        else:
            window = values[-self.period:]
            mean = np.tile(window, (horizon // self.period) + 1)[:horizon]
        latency = (time.perf_counter() - t0) * 1000
        return ForecastResult(
            mean=mean,
            lower=mean,
            upper=mean,
            latency_ms=latency,
            model_name=self.model_name,
        )
