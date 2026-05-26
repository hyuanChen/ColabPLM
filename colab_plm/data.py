"""Protein sequence data pipeline.

Handles three concerns that the original ColabSaprot code hardcoded for SaProt:
  * Spacing — ProtT5 expects whitespace-separated residues, ESM-2 does not.
  * Sanitization — non-standard residues (U Z O B) must collapse to X before
    tokenization or the tokenizer raises on unknown ids.
  * Length — long sequences are truncated to ``max_length`` to keep the LoRA
    fine-tune within Colab T4 memory.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(seq: str) -> str:
    seq = seq.upper().replace(" ", "")
    return "".join(c if c in _STANDARD_AA else "X" for c in seq)


class ProteinSequenceDataset(Dataset):
    """CSV-backed dataset where each row has ``protein`` and ``label`` columns."""

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 1024,
        is_t5: bool = False,
        is_regression: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        assert "protein" in self.df.columns and "label" in self.df.columns, (
            f"CSV {csv_path} must contain 'protein' and 'label' columns"
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_t5 = is_t5
        self.label_dtype = torch.float32 if is_regression else torch.long

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = sanitize_sequence(str(row["protein"]))
        if self.is_t5:
            seq = " ".join(list(seq))

        enc = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(row["label"], dtype=self.label_dtype)
        return item


def build_dataloaders(
    tokenizer,
    train_csv: str,
    val_csv: str | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    is_t5: bool = False,
    is_regression: bool = False,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader | None]:
    train_ds = ProteinSequenceDataset(
        train_csv, tokenizer, max_length, is_t5, is_regression
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_csv is not None:
        val_ds = ProteinSequenceDataset(
            val_csv, tokenizer, max_length, is_t5, is_regression
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader
