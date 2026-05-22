"""Aggregate raw commit timestamps into a monthly time series."""
from __future__ import annotations

import pandas as pd

MIN_MONTHS_FOR_FORECAST = 24


def to_monthly(commits: pd.DataFrame) -> pd.Series:
    """Return a pandas Series indexed by Period('M') with int commit counts.

    Missing months between min and max date are filled with 0.
    """
    if commits.empty:
        return pd.Series([], dtype="int64", name="commits",
                         index=pd.PeriodIndex([], freq="M"))

    dates = pd.to_datetime(commits["date"], utc=True).dt.tz_localize(None)
    months = dates.dt.to_period("M")
    counts = months.value_counts().sort_index()

    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    out = counts.reindex(full_range, fill_value=0).astype("int64")
    out.name = "commits"
    return out
