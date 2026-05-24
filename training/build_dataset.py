"""Build the GitHub multi-metric monthly dataset from a curated repo list.

Collects 4 metrics per repo per month: commits, prs_opened, issues_opened, stars_gained.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.github_fetch import RepoNotFoundError, fetch_repo_monthly_stats, parse_repo_url

logger = logging.getLogger(__name__)

MIN_MONTHS = 24
MAX_AVG_COMMITS_PER_MONTH = 300  # skip automated bot repos


def main(
    repos_yaml: Path = Path("training/repos.yaml"),
    output: Path = Path("data/training_dataset.parquet"),
    split: str = "train",
) -> None:
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN missing in environment")

    repos = yaml.safe_load(repos_yaml.read_text())[split]
    total = len(repos)
    rows: list[dict] = []
    kept = 0
    skipped_short = 0
    skipped_bot = 0

    for idx, entry in enumerate(repos, start=1):
        spec = parse_repo_url(entry)
        print(f"[{idx}/{total}] {entry}...", flush=True)
        try:
            df = fetch_repo_monthly_stats(spec.owner, spec.repo, token=token)
        except RepoNotFoundError:
            print(f"[{idx}/{total}] SKIP {entry}: not found", flush=True)
            continue
        except Exception as e:
            print(f"[{idx}/{total}] ERROR {entry}: {type(e).__name__}: {e}", flush=True)
            continue

        if len(df) < MIN_MONTHS:
            skipped_short += 1
            print(f"[{idx}/{total}] SKIP {entry}: only {len(df)} months", flush=True)
            continue

        avg_commits = df["commits"].mean()
        if avg_commits > MAX_AVG_COMMITS_PER_MONTH:
            skipped_bot += 1
            print(f"[{idx}/{total}] SKIP {entry}: bot ({avg_commits:.0f} commits/month)", flush=True)
            continue

        for i, (period, row) in enumerate(df.iterrows()):
            rows.append({
                "repo": entry,
                "month": period.to_timestamp(),
                "months_since_start": i,
                "commits": int(row["commits"]),
                "prs_opened": int(row["prs_opened"]),
                "issues_opened": int(row["issues_opened"]),
                "stars_gained": int(row["stars_gained"]),
            })
        kept += 1
        print(
            f"[{idx}/{total}] OK {entry}: {len(df)} months | "
            f"commits={int(df['commits'].sum())} prs={int(df['prs_opened'].sum())} "
            f"issues={int(df['issues_opened'].sum())} stars={int(df['stars_gained'].sum())}",
            flush=True,
        )

        if idx % 50 == 0:
            out_df = pd.DataFrame(rows)
            output.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(output, index=False)
            print(
                f"  [checkpoint] {len(out_df)} rows, {kept} repos kept "
                f"({skipped_short} short, {skipped_bot} bots) -> {output}",
                flush=True,
            )

    out_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["repo", "month", "months_since_start", "commits", "prs_opened", "issues_opened", "stars_gained"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)
    print(
        f"DONE: {len(out_df)} rows, {kept}/{total} repos kept "
        f"({skipped_short} short, {skipped_bot} bots) -> {output}",
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
