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
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    horizons = horizons or HORIZONS
    val_df = pd.read_parquet(val_parquet)

    forecasters = {
        "ours_ft": ChronosFineTuned(
            base_model_id=base_model, adapter_id=str(adapter_path), allow_fallback=False,
        ),
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
                s = pd.Series(train, name="commits")
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
    return (
        df.groupby(["horizon", "model"])[["smape", "mae", "latency_ms"]]
        .mean()
        .round(2)
        .reset_index()
    )


def main(val_parquet: Path, base_model: str, adapter_path: Path, results_md: Path):
    df = evaluate(val_parquet, base_model, adapter_path)
    summary = summarize(df)

    df.to_csv(results_md.with_suffix(".csv"), index=False)

    lines = [
        "# Evaluation Results\n",
        f"Validation repos: {df['repo'].nunique()}\n",
        "## SMAPE by horizon (lower is better)\n",
        "| Horizon | Ours (FT) | Chronos ZS | Naive |",
        "|---------|-----------|------------|-------|",
    ]
    for h in sorted(summary["horizon"].unique()):
        sub = summary[summary["horizon"] == h].set_index("model")["smape"]
        lines.append(
            f"| {h} mo | {sub.get('ours_ft', float('nan'))}% | "
            f"{sub.get('chronos_zs', float('nan'))}% | {sub.get('naive', float('nan'))}% |"
        )
    results_md.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--val-parquet", type=Path, default=Path("data/training_dataset_validation.parquet")
    )
    p.add_argument("--base-model", default="amazon/chronos-t5-small")
    p.add_argument("--adapter", type=Path, default=Path("models/chronos-github"))
    p.add_argument("--out", type=Path, default=Path("training/results.md"))
    args = p.parse_args()
    main(args.val_parquet, args.base_model, args.adapter, args.out)
