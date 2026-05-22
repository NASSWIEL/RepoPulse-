#!/usr/bin/env bash
set -euo pipefail

: "${HF_USERNAME:?HF_USERNAME must be set in environment}"

mkdir -p data models

python - <<PY
from huggingface_hub import snapshot_download
import os
username = os.environ["HF_USERNAME"]
snapshot_download(
    repo_id=f"{username}/github-monthly-commits",
    repo_type="dataset",
    local_dir="data",
)
snapshot_download(
    repo_id=f"{username}/chronos-github-commits",
    repo_type="model",
    local_dir="models/chronos-github",
)
PY

python training/evaluate.py \
    --val-parquet data/training_dataset_validation.parquet \
    --adapter models/chronos-github \
    --out training/results.md

echo "Done. See training/results.md"
