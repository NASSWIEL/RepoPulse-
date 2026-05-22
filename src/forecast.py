"""Forecasting models: naive seasonal baseline, Chronos zero-shot, Chronos fine-tuned."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

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
            window = values[-self.period :]
            mean = np.tile(window, (horizon // self.period) + 1)[:horizon]
        latency = (time.perf_counter() - t0) * 1000
        return ForecastResult(
            mean=mean,
            lower=mean,
            upper=mean,
            latency_ms=latency,
            model_name=self.model_name,
        )


class ChronosZeroShot:
    model_name = "chronos_zero_shot"

    def __init__(self, model_id: str = "amazon/chronos-t5-small", num_samples: int = 50):
        self.model_id = model_id
        self.num_samples = num_samples
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            import torch
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        return self._pipeline

    def forecast(self, series: pd.Series, horizon: int) -> ForecastResult:
        import torch

        t0 = time.perf_counter()
        pipe = self._load()
        context = torch.tensor(series.values, dtype=torch.float32)
        samples = pipe.predict(context, prediction_length=horizon, num_samples=self.num_samples)
        arr = samples.squeeze(0).numpy()
        mean = np.clip(arr.mean(axis=0), 0, None)
        lower = np.clip(np.quantile(arr, 0.1, axis=0), 0, None)
        upper = np.clip(np.quantile(arr, 0.9, axis=0), 0, None)
        latency = (time.perf_counter() - t0) * 1000
        return ForecastResult(
            mean=mean, lower=lower, upper=upper, latency_ms=latency, model_name=self.model_name
        )


class ChronosFineTuned(ChronosZeroShot):
    model_name = "chronos_finetuned"

    def __init__(
        self,
        base_model_id: str = "amazon/chronos-t5-small",
        adapter_id: str | None = None,
        num_samples: int = 50,
        allow_fallback: bool = True,
    ):
        super().__init__(model_id=base_model_id, num_samples=num_samples)
        self.adapter_id = adapter_id
        self.allow_fallback = allow_fallback
        self._loaded_with_adapter = False

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from chronos import ChronosPipeline

        try:
            pipe = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            if self.adapter_id:
                from peft import PeftModel

                pipe.model.model = PeftModel.from_pretrained(pipe.model.model, self.adapter_id)
                self._loaded_with_adapter = True
            self._pipeline = pipe
        except Exception:
            if not self.allow_fallback:
                raise
            self.model_name = "chronos_finetuned_fallback"
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        return self._pipeline
