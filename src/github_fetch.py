"""Fetch commit timestamps for a GitHub repository, with disk cache."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import urllib3


def _ssl_verify() -> bool:
    """Read GITHUB_VERIFY_SSL at call-time so dotenv-loaded values are honoured."""
    enabled = os.getenv("GITHUB_VERIFY_SSL", "true").lower() not in {"false", "0", "no"}
    if not enabled:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return enabled


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


def fetch_monthly_commits_fast(
    owner: str,
    repo: str,
    token: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    years_back: int = 5,
    max_pages: int = 30,
) -> pd.Series:
    """Fetch commits for the last `years_back` years, aggregate to monthly counts.

    Uses `since` date filtering so most lambda repos need only 5-15 API pages
    instead of 30-100+. Returns a pd.Series indexed by Period('M').
    """
    from src.aggregate import to_monthly

    spec = RepoSpec(owner, repo)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{spec.slug()}.monthly.parquet"

    if _cache_is_fresh(cache_path):
        df = pd.read_parquet(cache_path)
        if df.empty:
            return pd.Series(dtype="int64")
        return pd.Series(df["commits"].values, index=pd.PeriodIndex(df["month"], freq="M"), dtype="int64")

    since = datetime.now(timezone.utc) - timedelta(days=years_back * 365)

    session = requests.Session()
    session.verify = _ssl_verify()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/vnd.github+json"

    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params: dict = {"per_page": 100, "since": since.isoformat()}
    rows: list[dict] = []
    pages = 0

    while url and pages < max_pages:
        resp = session.get(url, params=params if pages == 0 else None, timeout=30)
        if resp.status_code == 404:
            raise RepoNotFoundError(f"{owner}/{repo} not found or private")
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            reset_at = datetime.fromtimestamp(reset, tz=timezone.utc)
            wait = max(0, reset - int(time.time())) + 5
            logger.warning("Rate limit hit on %s/%s. Sleeping %ds", owner, repo, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        for item in resp.json():
            rows.append({"date": item["commit"]["author"]["date"], "sha": item["sha"]})
        pages += 1
        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip().strip("<>")
        url = next_url

    if not rows:
        _save_monthly_cache(cache_path, pd.Series(dtype="int64"))
        return pd.Series(dtype="int64")

    df_commits = pd.DataFrame(rows)
    df_commits["date"] = pd.to_datetime(df_commits["date"], utc=True)
    s = to_monthly(df_commits)
    _save_monthly_cache(cache_path, s)
    return s


def _save_monthly_cache(cache_path: Path, s: pd.Series) -> None:
    if s.empty:
        df = pd.DataFrame({"month": pd.Series(dtype="object"), "commits": pd.Series(dtype="int64")})
    else:
        df = pd.DataFrame({"month": s.index.astype(str), "commits": s.values})
    df.to_parquet(cache_path, index=False)


def _paginate(
    session: requests.Session,
    url: str,
    params: dict,
    max_pages: int,
    since: datetime | None = None,
    date_field: str = "created_at",
) -> list[dict]:
    """Paginate a GitHub list endpoint, stopping early if items predate `since`."""
    rows: list[dict] = []
    pages = 0
    while url and pages < max_pages:
        resp = session.get(url, params=params if pages == 0 else None, timeout=30)
        if resp.status_code == 404:
            return rows
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            wait = max(0, reset - int(time.time())) + 5
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            break
        items = resp.json()
        if not items:
            break
        for item in items:
            rows.append(item)
        pages += 1
        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip().strip("<>")
        url = next_url
    return rows


def fetch_repo_monthly_stats(
    owner: str,
    repo: str,
    token: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    years_back: int = 5,
    max_pages: int = 30,
) -> pd.DataFrame:
    """Fetch commits, PRs opened, issues opened, stars gained — aggregated by month.

    Returns a DataFrame indexed by Period('M') with columns:
      commits, prs_opened, issues_opened, stars_gained
    Returns empty DataFrame on failure.
    Cache file: {slug}.stats.parquet (24h TTL).
    """
    spec = RepoSpec(owner, repo)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{spec.slug()}.stats.parquet"

    if _cache_is_fresh(cache_path):
        df = pd.read_parquet(cache_path)
        if df.empty:
            return df
        df.index = pd.PeriodIndex(df.index, freq="M")
        return df

    since = datetime.now(timezone.utc) - timedelta(days=years_back * 365)
    since_iso = since.isoformat()

    session = requests.Session()
    session.verify = _ssl_verify()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/vnd.github+json"

    base = f"{GITHUB_API}/repos/{owner}/{repo}"

    # 1. Commits
    commit_rows = _paginate(
        session,
        f"{base}/commits",
        {"per_page": 100, "since": since_iso},
        max_pages,
    )

    # 2. PRs (all states, sorted newest first so we can stop early)
    pr_rows = _paginate(
        session,
        f"{base}/pulls",
        {"per_page": 100, "state": "all", "sort": "created", "direction": "desc"},
        max_pages,
    )

    # 3. Issues (excludes PRs via is_pr check)
    issue_rows = _paginate(
        session,
        f"{base}/issues",
        {"per_page": 100, "state": "all", "sort": "created", "direction": "desc", "since": since_iso},
        max_pages,
    )

    # 4. Stars (Accept header for timestamps)
    session.headers["Accept"] = "application/vnd.github.star+json"
    star_rows = _paginate(
        session,
        f"{base}/stargazers",
        {"per_page": 100},
        max_pages,
    )
    session.headers["Accept"] = "application/vnd.github+json"

    # --- Aggregate to monthly ---
    def _to_period(date_str: str) -> pd.Period | None:
        try:
            return pd.Timestamp(date_str).to_period("M")
        except Exception:
            return None

    monthly: dict[pd.Period, dict] = {}

    def _inc(period: pd.Period | None, key: str) -> None:
        if period is None:
            return
        if period not in monthly:
            monthly[period] = {"commits": 0, "prs_opened": 0, "issues_opened": 0, "stars_gained": 0}
        monthly[period][key] += 1

    for c in commit_rows:
        try:
            _inc(_to_period(c["commit"]["author"]["date"]), "commits")
        except (KeyError, TypeError):
            pass

    for pr in pr_rows:
        dt = pr.get("created_at", "")
        if dt and pd.Timestamp(dt, tz="UTC") >= since:
            _inc(_to_period(dt), "prs_opened")

    for issue in issue_rows:
        if "pull_request" in issue:  # skip PRs listed in issues endpoint
            continue
        dt = issue.get("created_at", "")
        if dt and pd.Timestamp(dt, tz="UTC") >= since:
            _inc(_to_period(dt), "issues_opened")

    for star in star_rows:
        dt = star.get("starred_at", "")
        if dt and pd.Timestamp(dt, tz="UTC") >= since:
            _inc(_to_period(dt), "stars_gained")

    if not monthly:
        empty = pd.DataFrame(
            columns=["commits", "prs_opened", "issues_opened", "stars_gained"],
            dtype="int64",
        )
        empty.to_parquet(cache_path)
        return empty

    df = pd.DataFrame.from_dict(monthly, orient="index").sort_index().fillna(0).astype("int64")
    df.index = pd.PeriodIndex(df.index, freq="M")
    df_to_save = df.copy()
    df_to_save.index = df_to_save.index.astype(str)
    df_to_save.to_parquet(cache_path)
    return df


def fetch_commits(
    owner: str,
    repo: str,
    token: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    per_page: int = 100,
    max_pages: int = 500,
    wait_on_rate_limit: bool = False,
) -> pd.DataFrame:
    spec = RepoSpec(owner, repo)
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, spec)

    if _cache_is_fresh(path):
        logger.info("Cache hit for %s", spec.slug())
        return pd.read_parquet(path)

    session = requests.Session()
    session.verify = _ssl_verify()
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
            reset_at = datetime.fromtimestamp(reset, tz=timezone.utc)
            if wait_on_rate_limit:
                wait = max(0, reset - int(time.time())) + 5
                logger.warning(
                    "Rate limit hit on %s/%s. Sleeping %ds until %s",
                    owner,
                    repo,
                    wait,
                    reset_at.isoformat(),
                )
                time.sleep(wait)
                continue
            raise RateLimitError(reset_at)
        resp.raise_for_status()

        for item in resp.json():
            rows.append(
                {
                    "date": item["commit"]["author"]["date"],
                    "sha": item["sha"],
                }
            )

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
        df = pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns, UTC]"),
                "sha": pd.Series(dtype="object"),
            }
        )
    df.to_parquet(path, index=False)
    return df
