import numpy as np
import pandas as pd
import pytest
from src.metrics import smape, mae, backtest
from src.forecast import NaiveSeasonal


def test_smape_perfect_prediction_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert smape(y, y) == pytest.approx(0.0)


def test_smape_handles_zeros():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    assert smape(y_true, y_pred) == pytest.approx(0.0)


def test_smape_returns_percentage_0_to_200():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([15.0, 25.0, 35.0])
    val = smape(y_true, y_pred)
    assert 0 <= val <= 200


def test_mae_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 4.0])
    assert mae(y, p) == pytest.approx(2 / 3)


def test_backtest_returns_dataframe_with_expected_columns():
    idx = pd.period_range("2020-01", periods=36, freq="M")
    s = pd.Series(np.arange(36) + 5, index=idx, name="commits", dtype="int64")

    df = backtest(s, holdout=12, horizon=6, forecasters=[NaiveSeasonal(period=12)])
    assert set(df.columns) >= {"model", "smape", "mae", "latency_ms"}
    assert len(df) == 1
    assert df.iloc[0]["model"] == "naive_seasonal"


def test_backtest_does_not_leak_holdout_into_forecaster(monkeypatch):
    seen = {}

    class Spy:
        model_name = "spy"
        def forecast(self, series, horizon):
            seen["len"] = len(series)
            from src.forecast import ForecastResult
            return ForecastResult(
                mean=np.zeros(horizon), lower=np.zeros(horizon), upper=np.zeros(horizon),
                latency_ms=0.0, model_name="spy",
            )

    idx = pd.period_range("2020-01", periods=30, freq="M")
    s = pd.Series(range(30), index=idx, name="commits", dtype="int64")
    backtest(s, holdout=12, horizon=6, forecasters=[Spy()])
    assert seen["len"] == 18
