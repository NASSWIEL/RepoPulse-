"""Fetch commit timestamps for a GitHub repository, with disk cache."""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_TTL = timedelta(hours=24)
DEFAULT_CACHE_DIR = Path("data/cache")
GITHUB_API = "https://api.github.com"
_URL_PATTERN = re.compile(r"^(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


@dataclass(frozen=True)
class RepoSpec:
    owner: str
    repo: str

    def slug(self) -> str:
        return f"{self.owner}__{self.repo}"


def parse_repo_url(url: str) -> RepoSpec:
    url = url.strip()
    if "github.com" not in url and "/" not in url:
        raise ValueError(f"Invalid GitHub URL or owner/repo: {url!r}")
    if url.startswith(("http://", "https://")) and "github.com" not in url:
        raise ValueError(f"Only github.com URLs are supported, got: {url!r}")
    m = _URL_PATTERN.match(url)
    if not m:
        raise ValueError(f"Invalid GitHub URL or owner/repo: {url!r}")
    return RepoSpec(owner=m.group(1), repo=m.group(2))


class RepoNotFoundError(Exception):
    pass


class RateLimitError(Exception):
    def __init__(self, reset_at: datetime):
        self.reset_at = reset_at
        super().__init__(f"GitHub rate limit hit. Resets at {reset_at.isoformat()}")


def _cache_path(cache_dir: Path, spec: RepoSpec) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{spec.slug()}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age < CACHE_TTL


def fetch_commits(
    owner: str,
    repo: str,
    token: Optional[str] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    per_page: int = 100,
    max_pages: int = 500,
) -> pd.DataFrame:
    spec = RepoSpec(owner, repo)
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, spec)

    if _cache_is_fresh(path):
        logger.info("Cache hit for %s", spec.slug())
        return pd.read_parquet(path)

    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/vnd.github+json"

    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params = {"per_page": per_page}
    rows: list[dict] = []
    pages = 0

    while url and pages < max_pages:
        resp = session.get(url, params=params if pages == 0 else None, timeout=30)
        if resp.status_code == 404:
            raise RepoNotFoundError(f"{owner}/{repo} not found or private")
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            raise RateLimitError(datetime.fromtimestamp(reset, tz=timezone.utc))
        resp.raise_for_status()

        for item in resp.json():
            rows.append({
                "date": item["commit"]["author"]["date"],
                "sha": item["sha"],
            })

        pages += 1
        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip().strip("<>")
        url = next_url

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    else:
        df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]"), "sha": pd.Series(dtype="object")})
    df.to_parquet(path, index=False)
    return df
