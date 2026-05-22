import pandas as pd
import pytest
from src.aggregate import to_monthly, MIN_MONTHS_FOR_FORECAST


def _commits(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates), utc=True),
        "sha": [f"sha{i}" for i in range(len(dates))],
    })


def test_to_monthly_counts_per_month():
    df = _commits("2024-01-05", "2024-01-20", "2024-02-03")
    s = to_monthly(df)
    assert s.loc[pd.Period("2024-01", freq="M")] == 2
    assert s.loc[pd.Period("2024-02", freq="M")] == 1


def test_to_monthly_fills_missing_months_with_zero():
    df = _commits("2024-01-05", "2024-04-10")
    s = to_monthly(df)
    assert list(s.index) == [pd.Period(p, freq="M") for p in
                             ["2024-01", "2024-02", "2024-03", "2024-04"]]
    assert s.loc[pd.Period("2024-02", freq="M")] == 0
    assert s.loc[pd.Period("2024-03", freq="M")] == 0


def test_to_monthly_returns_int_series():
    df = _commits("2024-01-05")
    s = to_monthly(df)
    assert s.dtype.kind == "i"
    assert s.name == "commits"


def test_to_monthly_empty_returns_empty_series():
    df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]"),
                       "sha": pd.Series(dtype="object")})
    s = to_monthly(df)
    assert len(s) == 0
    assert s.name == "commits"


def test_min_months_constant_is_24():
    assert MIN_MONTHS_FOR_FORECAST == 24
