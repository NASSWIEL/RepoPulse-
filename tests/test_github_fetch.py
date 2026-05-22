import pytest
import pandas as pd
from src.github_fetch import parse_repo_url, RepoSpec


def test_parse_full_https_url():
    assert parse_repo_url("https://github.com/pytorch/pytorch") == RepoSpec("pytorch", "pytorch")


def test_parse_url_with_trailing_slash():
    assert parse_repo_url("https://github.com/facebook/react/") == RepoSpec("facebook", "react")


def test_parse_owner_repo_shorthand():
    assert parse_repo_url("pytorch/pytorch") == RepoSpec("pytorch", "pytorch")


def test_parse_url_with_dots_and_dashes():
    assert parse_repo_url("https://github.com/rust-lang/rust.git") == RepoSpec("rust-lang", "rust")


def test_parse_invalid_url_raises():
    with pytest.raises(ValueError, match="Invalid"):
        parse_repo_url("not a url")


def test_parse_non_github_url_raises():
    with pytest.raises(ValueError, match="github"):
        parse_repo_url("https://gitlab.com/foo/bar")


import responses
from datetime import datetime, timezone


@responses.activate
def test_fetch_commits_paginates_and_returns_dataframe(tmp_path):
    from src.github_fetch import fetch_commits

    page1 = [
        {"sha": "a" * 40, "commit": {"author": {"date": "2024-01-15T10:00:00Z"}}},
        {"sha": "b" * 40, "commit": {"author": {"date": "2024-01-20T11:00:00Z"}}},
    ]
    page2 = [
        {"sha": "c" * 40, "commit": {"author": {"date": "2024-02-01T08:00:00Z"}}},
    ]
    url = "https://api.github.com/repos/pytorch/pytorch/commits"
    responses.add(responses.GET, url, json=page1, status=200,
                  headers={"Link": f'<{url}?page=2>; rel="next"'})
    responses.add(responses.GET, url + "?page=2", json=page2, status=200)

    df = fetch_commits("pytorch", "pytorch", token="fake", cache_dir=tmp_path)

    assert len(df) == 3
    assert list(df.columns) == ["date", "sha"]
    assert df["date"].iloc[0].tzinfo is not None
    assert df["date"].iloc[0] == pd.Timestamp("2024-01-15T10:00:00Z")


@responses.activate
def test_fetch_commits_uses_cache_when_fresh(tmp_path):
    from src.github_fetch import fetch_commits

    cache_path = tmp_path / "pytorch__pytorch.parquet"
    cached = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "sha": ["x" * 40],
    })
    cached.to_parquet(cache_path)

    df = fetch_commits("pytorch", "pytorch", token="fake", cache_dir=tmp_path)
    assert len(df) == 1
    assert df["sha"].iloc[0] == "x" * 40
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_commits_raises_on_404(tmp_path):
    from src.github_fetch import fetch_commits, RepoNotFoundError

    responses.add(responses.GET,
                  "https://api.github.com/repos/foo/bar/commits",
                  status=404, json={"message": "Not Found"})

    with pytest.raises(RepoNotFoundError):
        fetch_commits("foo", "bar", token="fake", cache_dir=tmp_path)
