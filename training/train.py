"""Fine-tune Chronos-T5-small with LoRA on the GitHub monthly commits dataset.

Designed to run on a single T4 GPU (Colab free).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

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
        for _repo, group in df.sort_values(["repo", "month"]).groupby("repo"):
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
    from peft import LoraConfig, TaskType, get_peft_model

    pipe = ChronosPipeline.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=torch.float32,
    )
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


def compute_chronos_loss(
    pipe,
    context_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> torch.Tensor:
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

    train_ds = MonthlyCommitsWindowDataset(
        train_df,
        cfg.context_length,
        cfg.prediction_length,
        cfg.stride,
    )
    val_ds = MonthlyCommitsWindowDataset(
        val_df,
        cfg.context_length,
        cfg.prediction_length,
        cfg.stride,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    pipe = load_chronos_with_lora(cfg)
    pipe.model.model.train()

    optimizer = torch.optim.AdamW(
        [p for p in pipe.model.model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * cfg.warmup_ratio),
        total_steps,
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
    p.add_argument(
        "--train-parquet",
        type=Path,
        default=Path("data/training_dataset_train.parquet"),
    )
    p.add_argument(
        "--val-parquet",
        type=Path,
        default=Path("data/training_dataset_validation.parquet"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("models/chronos-github"))
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    cfg = TrainConfig(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )
    train(cfg)
