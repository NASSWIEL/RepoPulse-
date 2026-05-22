"""Build the GitHub monthly commits dataset from a curated repo list."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping
from pathlib import Path

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
            rows.append(
                {
                    "repo": repo,
                    "month": period.to_timestamp(),
                    "commits": int(val),
                    "months_since_start": i,
                }
            )
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
    total = len(repos)

    for idx, entry in enumerate(repos, start=1):
        spec = parse_repo_url(entry)
        print(f"[{idx}/{total}] fetching {entry}...", flush=True)
        try:
            commits = fetch_commits(
                spec.owner,
                spec.repo,
                token=token,
                wait_on_rate_limit=True,
            )
        except RepoNotFoundError:
            print(f"[{idx}/{total}] SKIP {entry}: not found", flush=True)
            continue
        except Exception as e:
            print(f"[{idx}/{total}] ERROR {entry}: {type(e).__name__}: {e}", flush=True)
            continue
        s = to_monthly(commits)
        if len(s) < 24:
            print(f"[{idx}/{total}] SKIP {entry}: only {len(s)} months", flush=True)
            continue
        series_map[entry] = s
        print(f"[{idx}/{total}] OK {entry}: {len(s)} months, {len(commits)} commits", flush=True)

        if idx % 25 == 0:
            partial = build_dataset_from_series(series_map)
            output.parent.mkdir(parents=True, exist_ok=True)
            partial.to_parquet(output, index=False)
            print(
                f"  [checkpoint] {len(partial)} rows, {len(series_map)} repos -> {output}",
                flush=True,
            )

    df = build_dataset_from_series(series_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(
        f"DONE: {len(df)} rows, {len(series_map)}/{total} repos kept -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], default="train")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or Path(f"data/training_dataset_{args.split}.parquet")
    main(output=out, split=args.split)
