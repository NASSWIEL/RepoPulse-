import pandas as pd
import pytest
from training.build_dataset import build_dataset_from_series


def test_build_dataset_from_series_produces_expected_schema():
    series_map = {
        "owner1/repoA": pd.Series(
            [1, 2, 3, 4],
            index=pd.period_range("2023-01", periods=4, freq="M"),
            dtype="int64", name="commits",
        ),
        "owner2/repoB": pd.Series(
            [5, 0, 7],
            index=pd.period_range("2024-06", periods=3, freq="M"),
            dtype="int64", name="commits",
        ),
    }
    df = build_dataset_from_series(series_map)
    assert set(df.columns) == {"repo", "month", "commits", "months_since_start"}
    assert df.shape == (7, 4)
    a = df[df["repo"] == "owner1/repoA"].sort_values("month")
    assert a["months_since_start"].tolist() == [0, 1, 2, 3]
    assert a["commits"].tolist() == [1, 2, 3, 4]
