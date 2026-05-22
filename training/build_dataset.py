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
