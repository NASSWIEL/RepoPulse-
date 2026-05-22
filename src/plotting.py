"""Plotly figure rendering for history + multiple forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from src.forecast import ForecastResult

_COLORS = {
    "chronos_finetuned": "#6366f1",
    "chronos_finetuned_fallback": "#a78bfa",
    "chronos_zero_shot": "#f59e0b",
    "naive_seasonal": "#94a3b8",
}


def _period_to_timestamp(idx: pd.PeriodIndex) -> pd.DatetimeIndex:
    return idx.to_timestamp()


def make_figure(
    history: pd.Series,
    forecasts: Sequence[ForecastResult],
    repo_label: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=_period_to_timestamp(history.index),
            y=history.values,
            mode="lines",
            name="History",
            line=dict(color="#475569", width=2),
        )
    )

    last_period = history.index[-1]
    horizon = len(forecasts[0].mean) if forecasts else 0
    future_idx = pd.period_range(last_period + 1, periods=horizon, freq="M")
    future_ts = _period_to_timestamp(future_idx)

    for f in forecasts:
        color = _COLORS.get(f.model_name, "#64748b")
        fig.add_trace(
            go.Scatter(
                x=future_ts,
                y=f.mean,
                mode="lines",
                name=f.model_name,
                line=dict(
                    color=color,
                    width=2,
                    dash="dot" if f.model_name == "naive_seasonal" else "solid",
                ),
            )
        )
        if "chronos_finetuned" in f.model_name:
            fig.add_trace(
                go.Scatter(
                    x=list(future_ts) + list(future_ts[::-1]),
                    y=list(f.upper) + list(f.lower[::-1]),
                    fill="toself",
                    fillcolor="rgba(99,102,241,0.15)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{f.model_name}_ci",
                )
            )

    fig.update_layout(
        title=f"Monthly commits forecast — {repo_label}",
        xaxis_title="Month",
        yaxis_title="Commits",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
