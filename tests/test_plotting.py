import numpy as np
import pandas as pd
from src.plotting import make_figure
from src.forecast import ForecastResult


def test_make_figure_returns_plotly_figure_with_expected_traces():
    idx = pd.period_range("2020-01", periods=24, freq="M")
    history = pd.Series(np.arange(24, dtype="int64"), index=idx, name="commits")

    forecasts = [
        ForecastResult(np.ones(6), np.zeros(6), 2 * np.ones(6), 10.0, "chronos_finetuned"),
        ForecastResult(np.ones(6) * 0.5, np.zeros(6), 1.5 * np.ones(6), 9.0, "chronos_zero_shot"),
        ForecastResult(np.full(6, 0.2), np.full(6, 0.2), np.full(6, 0.2), 0.1, "naive_seasonal"),
    ]

    fig = make_figure(history, forecasts, repo_label="pytorch/pytorch")
    trace_names = {t.name for t in fig.data}
    assert "History" in trace_names
    for f in forecasts:
        assert f.model_name in trace_names
    assert "pytorch/pytorch" in fig.layout.title.text
