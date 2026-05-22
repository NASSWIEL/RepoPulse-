"""Push the training dataset and the LoRA adapter to the Hugging Face Hub."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def push_dataset(api: HfApi, parquet: Path, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(parquet),
        path_in_repo=parquet.name,
        repo_id=repo_id,
        repo_type="dataset",
    )


def push_adapter(api: HfApi, adapter_dir: Path, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not (token and username):
        raise RuntimeError("HF_TOKEN and HF_USERNAME required in env")

    api = HfApi(token=token)
    dataset_repo = f"{username}/github-monthly-commits"
    adapter_repo = f"{username}/chronos-github-commits"

    for split in ("train", "validation"):
        parquet = Path(f"data/training_dataset_{split}.parquet")
        if parquet.exists():
            push_dataset(api, parquet, dataset_repo)
            print(f"Pushed {parquet.name} -> {dataset_repo}")
        else:
            print(f"Skipping {parquet} (not found)")

    adapter_dir = Path("models/chronos-github")
    if adapter_dir.exists():
        push_adapter(api, adapter_dir, adapter_repo)
        print(f"Pushed {adapter_dir}/ -> {adapter_repo}")
    else:
        print(f"Skipping {adapter_dir} (not found)")


if __name__ == "__main__":
    main()
