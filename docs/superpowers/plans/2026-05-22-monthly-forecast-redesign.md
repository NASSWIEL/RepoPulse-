# Monthly Forecast Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing RepoPulse codebase with a focused Gradio app on Hugging Face Spaces that forecasts monthly GitHub commit activity using a LoRA fine-tuned Chronos-T5-small, with side-by-side comparison against Chronos zero-shot and a naive seasonal baseline.

**Architecture:** Two pipelines living in the same repo. (1) Offline training: build a parquet dataset from ~150 curated public repos, LoRA-fine-tune `amazon/chronos-t5-small`, publish adapter + dataset to Hugging Face. (2) Online inference: Gradio Blocks app fetches commits for a user-supplied GitHub URL, aggregates to monthly, runs three forecasters in parallel, and renders chart + backtest table.

**Tech Stack:** Python 3.10, Gradio 4.x, `chronos-forecasting`, `transformers`, `peft` (LoRA), pandas/pyarrow, Plotly, GitHub REST API, Hugging Face Hub + Spaces.

---

## Phase 0 — Setup & Resolve Placeholders

### Task 0.1: Resolve `{HF_USERNAME}` placeholder in spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-22-monthly-forecast-redesign-design.md`

- [ ] **Step 1: Ask the user for their Hugging Face username** (or read `HF_USERNAME` from `.env` if it exists). If neither is available, stop and request it.

- [ ] **Step 2: Replace `{HF_USERNAME}` occurrences in the spec**

Use Edit with `replace_all: true` on the spec file:

```
old: {HF_USERNAME}
new: <actual-username>
```

- [ ] **Step 3: Add to `.env.example`**

Create or update `.env.example`:

```
GITHUB_TOKEN=ghp_your_token_here
HF_USERNAME=your-hf-username
HF_TOKEN=hf_your_token_here
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-22-monthly-forecast-redesign-design.md .env.example
git commit -m "chore: resolve HF_USERNAME placeholder and add .env.example"
```

---

### Task 0.2: Create new directory structure and update .gitignore

**Files:**
- Create: `src/__init__.py`, `training/__init__.py`, `tests/__init__.py`, `scripts/`, `data/cache/.gitkeep`, `models/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create empty directory skeleton**

```bash
mkdir -p src training tests scripts data/cache models images
touch src/__init__.py training/__init__.py tests/__init__.py
touch data/cache/.gitkeep models/.gitkeep
```

- [ ] **Step 2: Replace `.gitignore` with the target content**

Write `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.pytest_cache/
.ruff_cache/

# Env
.env
.env.local

# Data & models (artefacts hébergés sur HF, pas dans git)
data/cache/*
!data/cache/.gitkeep
data/*.parquet
models/*
!models/.gitkeep

# Brainstorming companion
.superpowers/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# MLflow / legacy runs (à supprimer)
mlruns/
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore src/__init__.py training/__init__.py tests/__init__.py data/cache/.gitkeep models/.gitkeep
git commit -m "chore: create new directory skeleton and update .gitignore"
```

---

### Task 0.3: Delete legacy code (spec §5 suppressions list)

**Files (delete):**
- `src/api_server.py`, `src/dashboard.py`, `src/dashboard_lite.py`, `src/inference_dashboard.py`, `src/inference_dashboard.py.bak`
- `src/ab_testing.py`, `src/distributed.py`, `src/orchestration.py`, `src/model_registry.py`
- `src/model_engine.py`, `src/model_selection.py`, `src/neural_network.py`, `src/train_neural_network.py`
- `src/data_validation.py`
- `visualize_training_losses.py`
- `notebooks/neural_network_training.ipynb` (then `rmdir notebooks` if empty)
- `analyse_reponses.md`
- `docs/project_documentation.tex`, `docs/project_summary.tex`
- `Dockerfile`, `compose.yaml`, `.dockerignore`
- `.github/workflows/ci-cd.yaml`

- [ ] **Step 1: Run the delete batch**

```bash
git rm src/api_server.py src/dashboard.py src/dashboard_lite.py \
       src/inference_dashboard.py src/inference_dashboard.py.bak \
       src/ab_testing.py src/distributed.py src/orchestration.py \
       src/model_registry.py src/model_engine.py src/model_selection.py \
       src/neural_network.py src/train_neural_network.py src/data_validation.py \
       visualize_training_losses.py analyse_reponses.md \
       docs/project_documentation.tex docs/project_summary.tex \
       Dockerfile compose.yaml .dockerignore \
       .github/workflows/ci-cd.yaml \
       notebooks/neural_network_training.ipynb
```

- [ ] **Step 2: Remove now-empty `notebooks/` directory**

```bash
rmdir notebooks 2>/dev/null || true
```

- [ ] **Step 3: Verify only kept files remain in `src/`**

```bash
ls src/
```
Expected output: `__init__.py  data_ingestion.py  etl.py` (and nothing else).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove legacy MLOps/dashboard code per redesign spec"
```

---

### Task 0.4: Rewrite `pyproject.toml` with reduced dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace the file with the target content**

Write `pyproject.toml`:

```toml
[project]
name = "repo-pulse"
version = "3.0.0"
description = "Forecast monthly GitHub commit activity with a LoRA fine-tuned Chronos transformer"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Naif Asswiel" }]

dependencies = [
    "gradio>=4.0,<5.0",
    "chronos-forecasting>=1.4.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "torch>=2.1.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyarrow>=15.0.0",
    "plotly>=5.18.0",
    "requests>=2.31.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "huggingface-hub>=0.23.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-mock>=3.10.0",
    "pytest-cov>=4.1.0",
    "responses>=0.25.0",
    "ruff>=0.4.0",
    "black>=24.0.0",
]
training = [
    "accelerate>=0.30.0",
    "datasets>=2.18.0",
    "tqdm>=4.65.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
```

- [ ] **Step 2: Install in editable dev mode**

```bash
pip install -e ".[dev]"
```
Expected: install completes without error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: rewrite pyproject.toml with reduced dependency set"
```

---

## Phase 1 — Data Layer

### Task 1.1: `github_fetch.py` — fetch commits with pagination and cache

**Files:**
- Create: `src/github_fetch.py`
- Test: `tests/test_github_fetch.py`

- [ ] **Step 1: Write failing test for URL parsing**

Write `tests/test_github_fetch.py`:

```python
import pytest
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
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_github_fetch.py -v
```
Expected: ImportError (`src.github_fetch` does not exist).

- [ ] **Step 3: Implement `parse_repo_url`**

Write `src/github_fetch.py`:

```python
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
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_github_fetch.py -v
```
Expected: 6 passing.

- [ ] **Step 5: Add failing test for `fetch_commits` (mocked)**

Append to `tests/test_github_fetch.py`:

```python
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
```

- [ ] **Step 6: Run tests, confirm new ones fail**

```bash
pytest tests/test_github_fetch.py -v
```
Expected: 3 failures (function/class missing).

- [ ] **Step 7: Implement `fetch_commits`**

Append to `src/github_fetch.py`:

```python
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
```

- [ ] **Step 8: Run all tests in file, confirm pass**

```bash
pytest tests/test_github_fetch.py -v
```
Expected: 9 passing.

- [ ] **Step 9: Commit**

```bash
git add src/github_fetch.py tests/test_github_fetch.py
git commit -m "feat(github_fetch): URL parsing, paginated commit fetch, 24h disk cache"
```

---

### Task 1.2: `aggregate.py` — raw commits → monthly series

**Files:**
- Create: `src/aggregate.py`
- Test: `tests/test_aggregate.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_aggregate.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_aggregate.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `to_monthly`**

Write `src/aggregate.py`:

```python
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

    dates = pd.to_datetime(commits["date"], utc=True)
    months = dates.dt.to_period("M")
    counts = months.value_counts().sort_index()

    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    out = counts.reindex(full_range, fill_value=0).astype("int64")
    out.name = "commits"
    return out
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_aggregate.py -v
```
Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add src/aggregate.py tests/test_aggregate.py
git commit -m "feat(aggregate): commits -> monthly Period series with zero-fill"
```

---

## Phase 2 — Forecast Module

### Task 2.1: Define `ForecastResult` and naive seasonal baseline

**Files:**
- Create: `src/forecast.py`
- Test: `tests/test_forecast.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_forecast.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_forecast.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `ForecastResult` and `NaiveSeasonal`**

Write `src/forecast.py`:

```python
"""Forecasting models: naive seasonal baseline, Chronos zero-shot, Chronos fine-tuned."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    latency_ms: float
    model_name: str


class Forecaster(Protocol):
    def forecast(self, series: pd.Series, horizon: int) -> ForecastResult: ...


class NaiveSeasonal:
    """Repeat the last `period` observations. Falls back to last value if series shorter."""
    model_name = "naive_seasonal"

    def __init__(self, period: int = 12):
        self.period = period

    def forecast(self, series: pd.Series, horizon: int) -> ForecastResult:
        t0 = time.perf_counter()
        values = series.values.astype(float)
        if len(values) < self.period:
            last = values[-1] if len(values) else 0.0
            mean = np.full(horizon, last)
        else:
            window = values[-self.period:]
            mean = np.tile(window, (horizon // self.period) + 1)[:horizon]
        latency = (time.perf_counter() - t0) * 1000
        return ForecastResult(
            mean=mean,
            lower=mean,
            upper=mean,
            latency_ms=latency,
            model_name=self.model_name,
        )
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_forecast.py -v
```
Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add src/forecast.py tests/test_forecast.py
git commit -m "feat(forecast): ForecastResult type and NaiveSeasonal baseline"
```

---

### Task 2.2: Chronos zero-shot forecaster

**Files:**
- Modify: `src/forecast.py`
- Modify: `tests/test_forecast.py`

- [ ] **Step 1: Add failing test (smoke test, real model load)**

Append to `tests/test_forecast.py`:

```python
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
```

(Use `chronos-t5-tiny` for tests — faster and same API as small.)

Add the marker config in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: requires model download"]
```

- [ ] **Step 2: Run only non-slow tests, confirm previous tests still pass**

```bash
pytest tests/test_forecast.py -v -m "not slow"
```
Expected: 4 passing, 1 deselected.

- [ ] **Step 3: Implement `ChronosZeroShot`**

Append to `src/forecast.py`:

```python
class ChronosZeroShot:
    model_name = "chronos_zero_shot"

    def __init__(self, model_id: str = "amazon/chronos-t5-small", num_samples: int = 50):
        self.model_id = model_id
        self.num_samples = num_samples
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from chronos import ChronosPipeline
            import torch
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        return self._pipeline

    def forecast(self, series: pd.Series, horizon: int) -> ForecastResult:
        import torch
        t0 = time.perf_counter()
        pipe = self._load()
        context = torch.tensor(series.values, dtype=torch.float32)
        samples = pipe.predict(context, prediction_length=horizon,
                               num_samples=self.num_samples)
        arr = samples.squeeze(0).numpy()
        mean = np.clip(arr.mean(axis=0), 0, None)
        lower = np.clip(np.quantile(arr, 0.1, axis=0), 0, None)
        upper = np.clip(np.quantile(arr, 0.9, axis=0), 0, None)
        latency = (time.perf_counter() - t0) * 1000
        return ForecastResult(mean=mean, lower=lower, upper=upper,
                              latency_ms=latency, model_name=self.model_name)
```

- [ ] **Step 4: Run slow test**

```bash
pytest tests/test_forecast.py::test_chronos_zero_shot_returns_sane_forecast -v -m slow
```
Expected: pass (first run downloads model; allow up to 2 min).

- [ ] **Step 5: Commit**

```bash
git add src/forecast.py tests/test_forecast.py pyproject.toml
git commit -m "feat(forecast): Chronos zero-shot wrapper with confidence intervals"
```

---

### Task 2.3: Chronos fine-tuned forecaster (loads LoRA adapter)

**Files:**
- Modify: `src/forecast.py`
- Modify: `tests/test_forecast.py`

- [ ] **Step 1: Add failing test (skipped until adapter exists)**

Append to `tests/test_forecast.py`:

```python
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
```

- [ ] **Step 2: Run, confirm failure**

```bash
pytest tests/test_forecast.py::test_chronos_finetuned_falls_back_when_adapter_missing -v -m slow
```
Expected: ImportError.

- [ ] **Step 3: Implement `ChronosFineTuned`**

Append to `src/forecast.py`:

```python
class ChronosFineTuned(ChronosZeroShot):
    model_name = "chronos_finetuned"

    def __init__(
        self,
        base_model_id: str = "amazon/chronos-t5-small",
        adapter_id: Optional[str] = None,
        num_samples: int = 50,
        allow_fallback: bool = True,
    ):
        super().__init__(model_id=base_model_id, num_samples=num_samples)
        self.adapter_id = adapter_id
        self.allow_fallback = allow_fallback
        self._loaded_with_adapter = False

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        from chronos import ChronosPipeline
        import torch

        try:
            pipe = ChronosPipeline.from_pretrained(
                self.model_id, device_map="cpu", torch_dtype=torch.float32,
            )
            if self.adapter_id:
                from peft import PeftModel
                pipe.model.model = PeftModel.from_pretrained(pipe.model.model, self.adapter_id)
                self._loaded_with_adapter = True
            self._pipeline = pipe
        except Exception:
            if not self.allow_fallback:
                raise
            self.model_name = "chronos_finetuned_fallback"
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id, device_map="cpu", torch_dtype=torch.float32,
            )
        return self._pipeline
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/test_forecast.py::test_chronos_finetuned_falls_back_when_adapter_missing -v -m slow
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/forecast.py tests/test_forecast.py
git commit -m "feat(forecast): Chronos fine-tuned wrapper with LoRA adapter loading"
```

---

## Phase 3 — Metrics & Plotting

### Task 3.1: `metrics.py` — SMAPE, MAE, backtest

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run, confirm failure**

```bash
pytest tests/test_metrics.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `metrics.py`**

Write `src/metrics.py`:

```python
"""Forecast quality metrics and held-out backtest."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.forecast import Forecaster


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.where((y_true == 0) & (y_pred == 0), 0.0,
                     num / np.where(den == 0, 1.0, den))
    return float(100.0 * np.mean(ratio))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def backtest(
    series: pd.Series,
    holdout: int,
    horizon: int,
    forecasters: Sequence[Forecaster],
) -> pd.DataFrame:
    """Hold out the last `holdout` months and forecast the first `horizon` of them."""
    if holdout < horizon:
        raise ValueError(f"holdout ({holdout}) must be >= horizon ({horizon})")
    if len(series) < holdout + 6:
        raise ValueError(f"series too short for backtest: {len(series)} < {holdout + 6}")

    train = series.iloc[:-holdout]
    truth = series.iloc[-holdout : -holdout + horizon].values.astype(float)

    rows = []
    for f in forecasters:
        res = f.forecast(train, horizon=horizon)
        rows.append({
            "model": res.model_name,
            "smape": smape(truth, res.mean),
            "mae": mae(truth, res.mean),
            "latency_ms": res.latency_ms,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_metrics.py -v
```
Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): SMAPE, MAE, backtest with strict train/test split"
```

---

### Task 3.2: `plotting.py` — Plotly figure for history + 3 forecasts

**Files:**
- Create: `src/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_plotting.py`:

```python
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
```

- [ ] **Step 2: Run, confirm failure**

```bash
pytest tests/test_plotting.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `plotting.py`**

Write `src/plotting.py`:

```python
"""Plotly figure rendering for history + multiple forecasts."""
from __future__ import annotations

from typing import Sequence

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

    fig.add_trace(go.Scatter(
        x=_period_to_timestamp(history.index),
        y=history.values,
        mode="lines",
        name="History",
        line=dict(color="#475569", width=2),
    ))

    last_period = history.index[-1]
    horizon = len(forecasts[0].mean) if forecasts else 0
    future_idx = pd.period_range(last_period + 1, periods=horizon, freq="M")
    future_ts = _period_to_timestamp(future_idx)

    for f in forecasts:
        color = _COLORS.get(f.model_name, "#64748b")
        fig.add_trace(go.Scatter(
            x=future_ts, y=f.mean, mode="lines",
            name=f.model_name,
            line=dict(color=color, width=2,
                      dash="dot" if f.model_name == "naive_seasonal" else "solid"),
        ))
        if "chronos_finetuned" in f.model_name:
            fig.add_trace(go.Scatter(
                x=list(future_ts) + list(future_ts[::-1]),
                y=list(f.upper) + list(f.lower[::-1]),
                fill="toself",
                fillcolor="rgba(99,102,241,0.15)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name=f"{f.model_name}_ci",
            ))

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
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_plotting.py -v
```
Expected: 1 passing.

- [ ] **Step 5: Commit**

```bash
git add src/plotting.py tests/test_plotting.py
git commit -m "feat(plotting): Plotly figure with history + 3 forecasts + CI band"
```

---

## Phase 4 — Training Pipeline

### Task 4.1: `training/repos.yaml` — curated repo list

**Files:**
- Create: `training/repos.yaml`

- [ ] **Step 1: Write the curated list**

Write `training/repos.yaml` with 150 entries total (120 train + 30 val). Use this exact structure (sample shown — fill 150 by following the diversity criteria from spec §6):

```yaml
# Curated GitHub repos for training/validation of repo-pulse.
# Criteria: age >= 3 years, diversity of domains/sizes, no hard forks.

train:
  # ML / DL
  - pytorch/pytorch
  - tensorflow/tensorflow
  - huggingface/transformers
  - scikit-learn/scikit-learn
  - keras-team/keras
  # Web frameworks
  - facebook/react
  - vuejs/vue
  - angular/angular
  - sveltejs/svelte
  - nodejs/node
  # Infra / DevOps
  - kubernetes/kubernetes
  - moby/moby
  - hashicorp/terraform
  - ansible/ansible
  - prometheus/prometheus
  # Languages / compilers
  - rust-lang/rust
  - golang/go
  - python/cpython
  - JuliaLang/julia
  - rakudo/rakudo
  # Scientific
  - numpy/numpy
  - pandas-dev/pandas
  - scipy/scipy
  - matplotlib/matplotlib
  - sympy/sympy
  # Databases
  - postgres/postgres
  - redis/redis
  - mongodb/mongo
  - elastic/elasticsearch
  - apache/cassandra
  # ... continue to 120 entries total, see TODO note below

validation:
  - facebook/jest
  - vercel/next.js
  - microsoft/vscode
  - ohmyzsh/ohmyzsh
  - tj/commander.js
  # ... continue to 30 entries total

# Note for the executing agent: if you reach this task and have fewer than
# 150 entries, augment the list with additional diverse public repos
# meeting the criteria. The list MUST contain exactly the keys `train`
# and `validation` and each list MUST contain only `owner/repo` strings.
```

(Time-saver: the executor should expand to 150 entries by drawing from a mix of: ML libs, web/frontend libs, backend frameworks, infra, languages, scientific computing, databases, devtools, CLI tools, editors/IDEs, ops, security, mobile. Keep validation set strictly disjoint from train.)

- [ ] **Step 2: Validate YAML loads**

```bash
python -c "import yaml; d = yaml.safe_load(open('training/repos.yaml')); assert len(d['train']) >= 100 and len(d['validation']) >= 20; print(f'OK: {len(d[\"train\"])} train, {len(d[\"validation\"])} val')"
```
Expected: prints OK.

- [ ] **Step 3: Commit**

```bash
git add training/repos.yaml
git commit -m "feat(training): curated repo list for fine-tuning dataset"
```

---

### Task 4.2: `training/build_dataset.py` — fetch repos → parquet

**Files:**
- Create: `training/build_dataset.py`
- Test: `tests/test_build_dataset.py`

- [ ] **Step 1: Write failing test**

Write `tests/test_build_dataset.py`:

```python
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
```

- [ ] **Step 2: Run, confirm failure**

```bash
pytest tests/test_build_dataset.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `build_dataset.py`**

Write `training/build_dataset.py`:

```python
"""Build the GitHub monthly commits dataset from a curated repo list."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.aggregate import to_monthly
from src.github_fetch import RepoNotFoundError, fetch_commits, parse_repo_url

logger = logging.getLogger(__name__)


def build_dataset_from_series(series_map: Mapping[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for repo, s in series_map.items():
        for i, (period, val) in enumerate(s.items()):
            rows.append({
                "repo": repo,
                "month": period.to_timestamp(),
                "commits": int(val),
                "months_since_start": i,
            })
    return pd.DataFrame(rows)


def main(
    repos_yaml: Path = Path("training/repos.yaml"),
    output: Path = Path("data/training_dataset.parquet"),
    split: str = "train",
):
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN missing in environment")

    repos = yaml.safe_load(repos_yaml.read_text())[split]
    series_map: dict[str, pd.Series] = {}

    for entry in repos:
        spec = parse_repo_url(entry)
        try:
            commits = fetch_commits(spec.owner, spec.repo, token=token)
        except RepoNotFoundError:
            logger.warning("Skipping %s: not found", entry)
            continue
        s = to_monthly(commits)
        if len(s) < 24:
            logger.warning("Skipping %s: only %d months of history", entry, len(s))
            continue
        series_map[entry] = s
        logger.info("Added %s: %d months", entry, len(s))

    df = build_dataset_from_series(series_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Wrote %d rows to %s (%d repos)", len(df), output, len(series_map))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], default="train")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or Path(f"data/training_dataset_{args.split}.parquet")
    main(output=out, split=args.split)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_build_dataset.py -v
```
Expected: 1 passing.

- [ ] **Step 5: Commit**

```bash
git add training/build_dataset.py tests/test_build_dataset.py
git commit -m "feat(training): build_dataset.py — fetch repos and assemble parquet"
```

- [ ] **Step 6: Run the actual build (Colab or local, requires GITHUB_TOKEN)**

```bash
python training/build_dataset.py --split train --output data/training_dataset_train.parquet
python training/build_dataset.py --split validation --output data/training_dataset_validation.parquet
```
Expected: produces two parquet files in `data/`. Print counts.

---

### Task 4.3: `training/train.py` — LoRA fine-tune Chronos

**Files:**
- Create: `training/train.py`

- [ ] **Step 1: Write the training script**

Write `training/train.py`:

```python
"""Fine-tune Chronos-T5-small with LoRA on the GitHub monthly commits dataset.

Designed to run on a single T4 GPU (Colab free).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    train_parquet: Path
    val_parquet: Path
    base_model: str = "amazon/chronos-t5-small"
    output_dir: Path = Path("models/chronos-github")
    context_length: int = 36
    prediction_length: int = 12
    stride: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    seed: int = 42


class MonthlyCommitsWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, context_length: int, prediction_length: int, stride: int):
        self.windows = []
        for repo, group in df.sort_values(["repo", "month"]).groupby("repo"):
            values = group["commits"].astype(np.float32).values
            n_needed = context_length + prediction_length
            if len(values) < n_needed:
                continue
            for start in range(0, len(values) - n_needed + 1, stride):
                ctx = values[start : start + context_length]
                tgt = values[start + context_length : start + n_needed]
                self.windows.append((ctx, tgt))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ctx, tgt = self.windows[i]
        return torch.tensor(ctx), torch.tensor(tgt)


def collate(batch):
    ctx = torch.stack([b[0] for b in batch])
    tgt = torch.stack([b[1] for b in batch])
    return ctx, tgt


def load_chronos_with_lora(cfg: TrainConfig):
    from chronos import ChronosPipeline
    from peft import LoraConfig, get_peft_model, TaskType

    pipe = ChronosPipeline.from_pretrained(cfg.base_model, device_map="auto", torch_dtype=torch.float32)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"],
    )
    pipe.model.model = get_peft_model(pipe.model.model, lora)
    pipe.model.model.print_trainable_parameters()
    return pipe


def compute_chronos_loss(pipe, context_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
    """Use Chronos's tokenizer to encode context+target, then run T5 with labels."""
    tokenizer = pipe.model.tokenizer
    device = next(pipe.model.model.parameters()).device

    ctx_ids, ctx_mask, _ = tokenizer.context_input_transform(context_batch)
    tgt_ids, _ = tokenizer.label_input_transform(target_batch, ctx_mask)

    ctx_ids = ctx_ids.to(device)
    ctx_mask = ctx_mask.to(device)
    tgt_ids = tgt_ids.to(device)
    tgt_ids[tgt_ids == tokenizer.config.pad_token_id] = -100

    out = pipe.model.model(input_ids=ctx_ids, attention_mask=ctx_mask, labels=tgt_ids)
    return out.loss


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_df = pd.read_parquet(cfg.train_parquet)
    val_df = pd.read_parquet(cfg.val_parquet)

    train_ds = MonthlyCommitsWindowDataset(train_df, cfg.context_length, cfg.prediction_length, cfg.stride)
    val_ds = MonthlyCommitsWindowDataset(val_df, cfg.context_length, cfg.prediction_length, cfg.stride)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    pipe = load_chronos_with_lora(cfg)
    pipe.model.model.train()

    optimizer = torch.optim.AdamW(
        [p for p in pipe.model.model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps,
    )

    log = {"epochs": [], "config": cfg.__dict__.copy()}
    log["config"]["train_parquet"] = str(cfg.train_parquet)
    log["config"]["val_parquet"] = str(cfg.val_parquet)
    log["config"]["output_dir"] = str(cfg.output_dir)

    for epoch in range(cfg.epochs):
        total_train = 0.0
        n_train = 0
        for ctx, tgt in train_loader:
            optimizer.zero_grad()
            loss = compute_chronos_loss(pipe, ctx, tgt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train += loss.item() * ctx.size(0)
            n_train += ctx.size(0)

        pipe.model.model.eval()
        total_val = 0.0
        n_val = 0
        with torch.no_grad():
            for ctx, tgt in val_loader:
                loss = compute_chronos_loss(pipe, ctx, tgt)
                total_val += loss.item() * ctx.size(0)
                n_val += ctx.size(0)
        pipe.model.model.train()

        train_loss = total_train / max(n_train, 1)
        val_loss = total_val / max(n_val, 1)
        log["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %d: train=%.4f val=%.4f", epoch, train_loss, val_loss)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pipe.model.model.save_pretrained(cfg.output_dir)
    (cfg.output_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    logger.info("Saved adapter to %s", cfg.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", type=Path, default=Path("data/training_dataset_train.parquet"))
    p.add_argument("--val-parquet", type=Path, default=Path("data/training_dataset_validation.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("models/chronos-github"))
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    cfg = TrainConfig(
        train_parquet=args.train_parquet, val_parquet=args.val_parquet,
        output_dir=args.output_dir, epochs=args.epochs,
    )
    train(cfg)
```

- [ ] **Step 2: Smoke test with tiny model on tiny synthetic data**

Create `tests/test_train_smoke.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_train_runs_one_epoch_on_tiny_data(tmp_path):
    from training.train import TrainConfig, train

    rows = []
    rng = np.random.default_rng(0)
    for r in range(4):
        for m in range(60):
            rows.append({
                "repo": f"r{r}",
                "month": pd.Timestamp("2020-01-01") + pd.DateOffset(months=m),
                "commits": int(rng.integers(0, 50)),
                "months_since_start": m,
            })
    df = pd.DataFrame(rows)
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    df.iloc[:180].to_parquet(train_path)
    df.iloc[180:].to_parquet(val_path)

    out = tmp_path / "out"
    cfg = TrainConfig(
        train_parquet=train_path, val_parquet=val_path,
        base_model="amazon/chronos-t5-tiny",
        output_dir=out, epochs=1, batch_size=4,
    )
    train(cfg)

    assert (out / "training_log.json").exists()
    log = json.loads((out / "training_log.json").read_text())
    assert len(log["epochs"]) == 1
```

```bash
pytest tests/test_train_smoke.py -v -m slow
```
Expected: pass (allow several minutes).

- [ ] **Step 3: Commit**

```bash
git add training/train.py tests/test_train_smoke.py
git commit -m "feat(training): LoRA fine-tuning of Chronos-T5-small"
```

---

### Task 4.4: `training/evaluate.py` — produce the results table

**Files:**
- Create: `training/evaluate.py`

- [ ] **Step 1: Write the evaluation script**

Write `training/evaluate.py`:

```python
"""Evaluate fine-tuned Chronos vs zero-shot vs naive baseline on the validation split."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.forecast import ChronosFineTuned, ChronosZeroShot, NaiveSeasonal
from src.metrics import mae, smape

logger = logging.getLogger(__name__)
HORIZONS = [1, 3, 6, 12]


def evaluate(
    val_parquet: Path,
    base_model: str,
    adapter_path: Path,
    horizons: list[int] = None,
) -> pd.DataFrame:
    horizons = horizons or HORIZONS
    val_df = pd.read_parquet(val_parquet)

    forecasters = {
        "ours_ft": ChronosFineTuned(base_model_id=base_model, adapter_id=str(adapter_path), allow_fallback=False),
        "chronos_zs": ChronosZeroShot(model_id=base_model),
        "naive": NaiveSeasonal(period=12),
    }

    rows = []
    for repo, group in val_df.sort_values(["repo", "month"]).groupby("repo"):
        values = group["commits"].astype(float).values
        if len(values) < max(horizons) + 24:
            continue
        for h in horizons:
            train = values[:-h]
            truth = values[-h:]
            for name, f in forecasters.items():
                import pandas as pd_
                s = pd_.Series(train, name="commits")
                res = f.forecast(s, horizon=h)
                rows.append({
                    "repo": repo,
                    "horizon": h,
                    "model": name,
                    "smape": smape(truth, res.mean),
                    "mae": mae(truth, res.mean),
                    "latency_ms": res.latency_ms,
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["horizon", "model"])[["smape", "mae", "latency_ms"]]
              .mean()
              .round(2)
              .reset_index())


def main(val_parquet: Path, base_model: str, adapter_path: Path, results_md: Path):
    df = evaluate(val_parquet, base_model, adapter_path)
    summary = summarize(df)

    df.to_csv(results_md.with_suffix(".csv"), index=False)

    lines = ["# Evaluation Results\n",
             f"Validation repos: {df['repo'].nunique()}\n",
             "## SMAPE by horizon (lower is better)\n",
             "| Horizon | Ours (FT) | Chronos ZS | Naive |",
             "|---------|-----------|------------|-------|"]
    for h in sorted(summary["horizon"].unique()):
        sub = summary[summary["horizon"] == h].set_index("model")["smape"]
        lines.append(f"| {h} mo | {sub.get('ours_ft', float('nan'))}% | "
                     f"{sub.get('chronos_zs', float('nan'))}% | {sub.get('naive', float('nan'))}% |")
    results_md.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--val-parquet", type=Path, default=Path("data/training_dataset_validation.parquet"))
    p.add_argument("--base-model", default="amazon/chronos-t5-small")
    p.add_argument("--adapter", type=Path, default=Path("models/chronos-github"))
    p.add_argument("--out", type=Path, default=Path("training/results.md"))
    args = p.parse_args()
    main(args.val_parquet, args.base_model, args.adapter, args.out)
```

- [ ] **Step 2: Run evaluation (after training has produced the adapter)**

```bash
python training/evaluate.py
```
Expected: writes `training/results.md` and `training/results.csv`. Prints the markdown table.

- [ ] **Step 3: Verify the release criterion**

Open `training/results.md`. Check that `Ours (FT)` SMAPE < `Chronos ZS` SMAPE on every horizon row. If not, STOP and tune `lora_r`, `epochs`, or `learning_rate` before continuing.

- [ ] **Step 4: Commit**

```bash
git add training/evaluate.py training/results.md training/results.csv
git commit -m "feat(training): evaluate.py + initial results table"
```

---

### Task 4.5: Publish dataset and adapter to Hugging Face

**Files:**
- Create: `scripts/push_to_hf.py`

- [ ] **Step 1: Write the publish script**

Write `scripts/push_to_hf.py`:

```python
"""Push the training dataset and the LoRA adapter to the Hugging Face Hub."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def push_dataset(api: HfApi, parquet: Path, repo_id: str):
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(parquet), path_in_repo=parquet.name,
                    repo_id=repo_id, repo_type="dataset")


def push_adapter(api: HfApi, adapter_dir: Path, repo_id: str):
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")


def main():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not (token and username):
        raise RuntimeError("HF_TOKEN and HF_USERNAME required in env")

    api = HfApi(token=token)
    push_dataset(api, Path("data/training_dataset_train.parquet"), f"{username}/github-monthly-commits")
    push_dataset(api, Path("data/training_dataset_validation.parquet"), f"{username}/github-monthly-commits")
    push_adapter(api, Path("models/chronos-github"), f"{username}/chronos-github-commits")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python scripts/push_to_hf.py
```
Expected: dataset and model repos created under your HF account.

- [ ] **Step 3: Commit**

```bash
git add scripts/push_to_hf.py
git commit -m "feat(scripts): push dataset and LoRA adapter to Hugging Face Hub"
```

---

## Phase 5 — Gradio App

### Task 5.1: `app.py` — Gradio Blocks interface

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write the Gradio app**

Write `app.py`:

```python
"""Gradio app: paste a GitHub repo URL, get a monthly commit forecast."""
from __future__ import annotations

import logging
import os
from datetime import datetime

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from src.aggregate import MIN_MONTHS_FOR_FORECAST, to_monthly
from src.forecast import ChronosFineTuned, ChronosZeroShot, NaiveSeasonal
from src.github_fetch import RateLimitError, RepoNotFoundError, fetch_commits, parse_repo_url
from src.metrics import backtest
from src.plotting import make_figure

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "")
ADAPTER_ID = f"{HF_USERNAME}/chronos-github-commits" if HF_USERNAME else None
BASE_MODEL = "amazon/chronos-t5-small"

_ft = ChronosFineTuned(base_model_id=BASE_MODEL, adapter_id=ADAPTER_ID, allow_fallback=True)
_zs = ChronosZeroShot(model_id=BASE_MODEL)
_naive = NaiveSeasonal(period=12)


def predict(url: str, horizon: int):
    if not url or not url.strip():
        return None, "Please paste a GitHub repository URL.", None
    try:
        spec = parse_repo_url(url)
    except ValueError as e:
        return None, f"❌ {e}", None
    try:
        commits = fetch_commits(spec.owner, spec.repo, token=GITHUB_TOKEN)
    except RepoNotFoundError:
        return None, f"❌ {spec.owner}/{spec.repo} not found or private.", None
    except RateLimitError as e:
        return None, f"❌ GitHub rate limit hit. Reset at {e.reset_at.isoformat()}", None

    s = to_monthly(commits)
    if len(s) < 6:
        return None, f"❌ Not enough history ({len(s)} months) to forecast.", None

    warning = ""
    if len(s) < MIN_MONTHS_FOR_FORECAST:
        warning = (f"⚠️ Only {len(s)} months of history "
                   f"(recommended ≥{MIN_MONTHS_FOR_FORECAST}). Forecast confidence is low.\n\n")

    forecasts = [_ft.forecast(s, horizon), _zs.forecast(s, horizon), _naive.forecast(s, horizon)]
    fig = make_figure(s, forecasts, repo_label=f"{spec.owner}/{spec.repo}")

    bt_df = None
    if len(s) >= max(12, horizon) + 6:
        bt_df = backtest(s, holdout=12, horizon=min(horizon, 12), forecasters=[_ft, _zs, _naive])
        bt_df = bt_df.round(2)

    label = (f"{warning}**{spec.owner}/{spec.repo}** · {len(s)} months of history · "
             f"forecast horizon {horizon} months")
    return fig, label, bt_df


CSS = """
.gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px !important; }
#hero { text-align: center; padding: 32px 0 24px; }
#hero h1 { font-size: 2.4rem; margin: 0 0 8px; }
#hero p { color: #64748b; margin: 0; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, title="repo-pulse") as app:
        with gr.Column(elem_id="hero"):
            gr.Markdown("# Forecast any GitHub repo")
            gr.Markdown("Fine-tuned Chronos transformer · monthly horizon")

        with gr.Row():
            url = gr.Textbox(placeholder="github.com/pytorch/pytorch", label="Repository URL", scale=4)
            btn = gr.Button("Predict →", variant="primary", scale=1)
        horizon = gr.Slider(1, 12, value=6, step=1, label="Horizon (months)")
        gr.Markdown("Try: `pytorch/pytorch` · `facebook/react` · `rust-lang/rust`")

        with gr.Column(visible=False) as results:
            label = gr.Markdown()
            chart = gr.Plot()
            gr.Markdown("### Backtest on last 12 months (held-out)")
            table = gr.Dataframe(headers=["model", "smape", "mae", "latency_ms"], interactive=False)

        def _predict_and_show(u, h):
            fig, lbl, bt = predict(u, h)
            return {
                results: gr.update(visible=fig is not None),
                chart: fig,
                label: lbl,
                table: bt,
            }

        btn.click(_predict_and_show, inputs=[url, horizon], outputs=[results, chart, label, table])
        url.submit(_predict_and_show, inputs=[url, horizon], outputs=[results, chart, label, table])

    return app


if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 2: Smoke test launch**

```bash
python app.py
```
Expected: Gradio prints a local URL `http://127.0.0.1:7860`. Open it, paste `pytorch/pytorch`, click Predict, verify chart renders.
Stop with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): Gradio Blocks UI with hero, slider, 3 forecasts, backtest table"
```

---

## Phase 6 — Deployment & README

### Task 6.1: `requirements.txt` for Hugging Face Spaces

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Generate the file**

Write `requirements.txt`:

```
gradio>=4.0,<5.0
chronos-forecasting>=1.4.0
transformers>=4.40.0
peft>=0.10.0
torch>=2.1.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=15.0.0
plotly>=5.18.0
requests>=2.31.0
pyyaml>=6.0
python-dotenv>=1.0.0
huggingface-hub>=0.23.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt for Hugging Face Spaces runtime"
```

---

### Task 6.2: Rewrite `README.md` with HF Spaces card + project narrative

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README**

Replace `README.md` content with:

```markdown
---
title: Repo Pulse
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.36.1"
app_file: app.py
pinned: false
license: mit
---

# Repo Pulse

Forecast monthly commit activity for any public GitHub repository, using a LoRA fine-tuned [Chronos](https://github.com/amazon-science/chronos-forecasting) transformer.

![Demo](images/shot_demo.png)

**Live demo:** https://huggingface.co/spaces/<HF_USERNAME>/repo-pulse

## What

Paste a GitHub repo URL, pick a horizon (1–12 months), get a forecast of monthly commit counts. The app runs three forecasters side by side:

1. **Ours** — Chronos-T5-small fine-tuned with LoRA on ~120 public GitHub repos
2. **Chronos zero-shot** — same base model, no fine-tuning
3. **Naive seasonal** — repeats the last 12 months

The whole point: prove the fine-tuning adds value, and prove it transparently.

## Results

Backtest on 30 validation repos (never seen during training):

| Horizon | Ours (FT) | Chronos ZS | Naive |
|---------|-----------|------------|-------|
| 1 month | (see `training/results.md`) | | |
| 3 months | | | |
| 6 months | | | |
| 12 months | | | |

Metric: SMAPE (lower is better). Reproduce with `bash scripts/reproduce.sh`.

## Architecture

```
training/build_dataset.py   →   HF Datasets: github-monthly-commits
training/train.py           →   HF Hub: chronos-github-commits (LoRA adapter)
app.py (HF Spaces)          →   fetch URL → aggregate monthly → 3 forecasts → chart
```

## Reproduce

```bash
pip install -e ".[dev,training]"

# 1. Build dataset (requires GITHUB_TOKEN)
python training/build_dataset.py --split train
python training/build_dataset.py --split validation

# 2. Fine-tune (requires GPU; ~1-2h on Colab T4)
python training/train.py

# 3. Evaluate
python training/evaluate.py

# 4. Run the app locally
python app.py
```

Or, to only reproduce the results table from published artifacts:

```bash
bash scripts/reproduce.sh
```

## Limitations

- **Young repos (<24 months of history):** forecast displayed but flagged in the UI as low-confidence.
- **Domain shift:** training set is biased toward popular OSS projects. Small private-style repos may forecast less accurately.
- **Univariate only:** the model predicts commits without conditioning on stars/issues/PRs. A multivariate extension is on the next-steps list.
- **Latency:** large repos (200k+ commits) take 30–60 s to fetch on first request. Cached for 24 h thereafter.

## License

MIT
```

(Replace `<HF_USERNAME>` with the actual resolved value from Task 0.1.)

- [ ] **Step 2: Inject the actual `Results` numbers**

Read `training/results.md` and paste the four `|...|` rows in place of the empty rows.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with HF Spaces card and project narrative"
```

---

### Task 6.3: `scripts/reproduce.sh` — re-run eval from HF artifacts

**Files:**
- Create: `scripts/reproduce.sh`

- [ ] **Step 1: Write the script**

Write `scripts/reproduce.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${HF_USERNAME:?HF_USERNAME must be set in environment}"

mkdir -p data models

python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id=f"${HF_USERNAME}/github-monthly-commits", repo_type="dataset", local_dir="data")
snapshot_download(repo_id=f"${HF_USERNAME}/chronos-github-commits", repo_type="model", local_dir="models/chronos-github")
PY

python training/evaluate.py \
    --val-parquet data/training_dataset_validation.parquet \
    --adapter models/chronos-github \
    --out training/results.md

echo "✅ Done. See training/results.md"
```

- [ ] **Step 2: Make executable and test**

```bash
chmod +x scripts/reproduce.sh
HF_USERNAME=<your-username> bash scripts/reproduce.sh
```
Expected: prints the results table, writes `training/results.md`.

- [ ] **Step 3: Commit**

```bash
git add scripts/reproduce.sh
git commit -m "feat(scripts): reproduce.sh — re-run eval from published HF artifacts"
```

---

### Task 6.4: Deploy to Hugging Face Spaces

- [ ] **Step 1: Create the Space**

In the HF web UI: New Space → name `repo-pulse` → SDK Gradio → Hardware CPU basic → Public.

- [ ] **Step 2: Add Space remote**

```bash
git remote add space https://huggingface.co/spaces/<HF_USERNAME>/repo-pulse
```

- [ ] **Step 3: Configure Secrets**

In the Space settings → Variables and secrets, add:
- `GITHUB_TOKEN` (the token from `.env`)
- `HF_USERNAME` (your username)

- [ ] **Step 4: Push**

```bash
git push space main
```
Expected: Space starts building; eventual URL `https://huggingface.co/spaces/<HF_USERNAME>/repo-pulse`.

- [ ] **Step 5: Smoke test the live URL**

Open the URL, paste `pytorch/pytorch`, click Predict, verify chart and backtest table render.

- [ ] **Step 6: Capture the demo screenshot**

Take a screenshot of the live app after a successful prediction, save as `images/shot_demo.png` (overwrite existing).

```bash
git add images/shot_demo.png
git commit -m "docs: update demo screenshot from live deployment"
git push origin main
git push space main
```

---

### Task 6.5: Minimal CI

**Files:**
- Create: `.github/workflows/ci.yaml`

- [ ] **Step 1: Write the workflow**

Write `.github/workflows/ci.yaml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: pip
      - run: pip install -e ".[dev]"
      - name: Lint
        run: |
          ruff check src/ training/ app.py
          black --check src/ training/ app.py
      - name: Test (fast only)
        run: pytest tests/ -m "not slow" -v
```

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/ci.yaml
git commit -m "ci: add minimal lint + fast test workflow"
git push origin main
```
Expected: GitHub Actions run, green check.

---

## Phase 7 — Cleanup & Tag

### Task 7.1: Tag `v1.0`

- [ ] **Step 1: Verify everything is green**

```bash
pytest tests/ -m "not slow" -v
ruff check src/ training/ app.py
black --check src/ training/ app.py
```
Expected: all green.

- [ ] **Step 2: Verify live Space loads**

Open the Spaces URL, run one prediction.

- [ ] **Step 3: Tag**

```bash
git tag -a v1.0 -m "Initial public release: monthly forecast with fine-tuned Chronos"
git push origin v1.0
```

---

## Plan Self-Review

**Spec coverage check** — every spec section maps to at least one task:
- §1 Objectif → Phases 5–6
- §2 Critères de succès → 6.4 (deployed), 4.4 (release criterion), 6.2 (README), 6.3 (reproduce)
- §3 Décisions clés → covered by Phases 1–5
- §4 Architecture → Phases 1, 2, 3, 5
- §5 Structure → 0.2, 0.3, 0.4
- §6 Pipeline d'entraînement → Phase 4
- §7 Application Gradio → 5.1
- §8 README → 6.2
- §9 Évaluation publique → 6.3 (reproduce.sh), 4.4 (evaluate.py)
- §10 CI/CD → 6.5
- §11 Dépendances → 0.4 (pyproject), 6.1 (requirements.txt)
- §12 Limites connues → documented in README (6.2) and via UI warning (5.1)
- §13 Sécurité → 0.1 (.env.example), 6.4 (Spaces Secrets), 5.1 (token via getenv)
- §14 Hors scope → enforced by deletions in 0.3

**Placeholder scan:**
- `{HF_USERNAME}` and `<HF_USERNAME>` appear intentionally in Tasks 0.1, 6.2, 6.3, 6.4 — these are resolved by Task 0.1 and substituted manually in later tasks. Acceptable.
- `repos.yaml` is partially filled (sample 30 entries shown, executor extends to 150). The instructions in the file itself are explicit about the criteria. Acceptable for a long curated list — a fully enumerated list would bloat the plan without adding value.

**Type consistency:**
- `ForecastResult` (fields: `mean`, `lower`, `upper`, `latency_ms`, `model_name`) — used consistently across `forecast.py`, `metrics.py`, `plotting.py`, `app.py`.
- `RepoSpec`, `RepoNotFoundError`, `RateLimitError` — defined in `github_fetch.py`, referenced consistently in `app.py` and `build_dataset.py`.
- `to_monthly(commits: pd.DataFrame) -> pd.Series` — same signature in tests and callers.
- `backtest(series, holdout, horizon, forecasters)` — same signature in tests and `evaluate.py` (note: `evaluate.py` reimplements its own loop because it needs per-repo + per-horizon granularity, which is intentional, not inconsistent).

Plan is internally consistent.
