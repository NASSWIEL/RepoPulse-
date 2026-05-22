import numpy as np
import pandas as pd
import pytest
from src.forecast import ForecastResult, NaiveSeasonal


def _series(values: list[int]) -> pd.Series:
    idx = pd.period_range("2020-01", periods=len(values), freq="M")
    return pd.Series(values, index=idx, name="commits", dtype="int64")


def test_forecast_result_shapes():
    res = ForecastResult(
        mean=np.array([1.0, 2.0, 3.0]),
        lower=np.array([0.5, 1.5, 2.5]),
        upper=np.array([1.5, 2.5, 3.5]),
        latency_ms=12.3,
        model_name="naive",
    )
    assert len(res.mean) == 3
    assert res.latency_ms > 0


def test_naive_seasonal_repeats_last_12_months():
    values = list(range(1, 25))
    s = _series(values)
    forecaster = NaiveSeasonal(period=12)
    res = forecaster.forecast(s, horizon=12)
    assert res.mean.tolist() == values[-12:]
    assert res.model_name == "naive_seasonal"


def test_naive_seasonal_horizon_longer_than_period_wraps():
    s = _series(list(range(1, 25)))
    res = NaiveSeasonal(period=12).forecast(s, horizon=15)
    assert len(res.mean) == 15
    assert res.mean[12] == res.mean[0]


def test_naive_seasonal_falls_back_when_too_short():
    s = _series([5, 6, 7])
    res = NaiveSeasonal(period=12).forecast(s, horizon=6)
    assert (res.mean == 7).all()


@pytest.mark.slow
def test_chronos_zero_shot_returns_sane_forecast():
    from src.forecast import ChronosZeroShot

    s = _series([10, 12, 8, 15, 20, 14, 9, 11, 13, 16, 18, 12,
                 11, 13, 9, 16, 21, 15, 10, 12, 14, 17, 19, 13])
    forecaster = ChronosZeroShot(model_id="amazon/chronos-t5-tiny")
    res = forecaster.forecast(s, horizon=6)

    assert len(res.mean) == 6
    assert (res.mean >= 0).all()
    assert (res.upper >= res.mean).all()
    assert (res.lower <= res.mean).all()
    assert res.model_name == "chronos_zero_shot"


@pytest.mark.slow
def test_chronos_finetuned_falls_back_when_adapter_missing():
    from src.forecast import ChronosFineTuned

    s = _series(list(range(1, 25)))
    forecaster = ChronosFineTuned(
        base_model_id="amazon/chronos-t5-tiny",
        adapter_id="nonexistent/adapter-does-not-exist",
        allow_fallback=True,
    )
    res = forecaster.forecast(s, horizon=6)
    assert res.model_name in ("chronos_finetuned", "chronos_finetuned_fallback")
    assert len(res.mean) == 6
