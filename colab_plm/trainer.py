"""Minimal training/eval/inference loops for ColabPLM."""
from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from .loader import BackboneBundle, ColabPLMLoader
from .data import build_dataloaders, sanitize_sequence
from .adapter import inject_lora_adapter


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader: DataLoader, is_regression: bool = False):
    model.eval()
    device = next(model.parameters()).device
    preds: List[float] = []
    golds: List[float] = []
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += float(out.loss)
        if is_regression:
            preds.extend(out.logits.squeeze(-1).cpu().tolist())
        else:
            preds.extend(out.logits.argmax(-1).cpu().tolist())
        golds.extend(batch["labels"].cpu().tolist())
    if is_regression:
        metric = {"mse": mean_squared_error(golds, preds)}
    else:
        metric = {
            "acc": accuracy_score(golds, preds),
            "f1": f1_score(golds, preds, average="macro"),
        }
    metric["loss"] = total_loss / max(len(loader), 1)
    return metric


def train_colab_plm(
    model_name: str,
    train_csv: str,
    val_csv: str | None,
    num_labels: int,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-4,
    rank: int = 8,
    alpha: int = 16,
    max_length: int = 512,
    is_regression: bool = False,
    target_modules: Iterable[str] | None = None,
    cache_dir: str | None = None,
    fp16: bool = True,
    log_every: int = 5,
) -> Tuple[torch.nn.Module, BackboneBundle, dict]:
    """One-shot orchestrator: load → inject LoRA → fine-tune → save."""
    device = _device()
    print(f"[ColabPLM] device={device}")

    bundle = ColabPLMLoader.load_backbone(
        model_name, num_labels=num_labels,
        is_regression=is_regression, cache_dir=cache_dir,
    )
    print(f"[ColabPLM] family={bundle.family} hidden_size={bundle.hidden_size}")

    peft_model = inject_lora_adapter(
        bundle.model, target_modules=target_modules, rank=rank, alpha=alpha,
    ).to(device)

    train_loader, val_loader = build_dataloaders(
        bundle.tokenizer, train_csv, val_csv,
        batch_size=batch_size, max_length=max_length,
        is_t5=bundle.is_t5, is_regression=is_regression,
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and device.type == "cuda")

    history = {"train_loss": [], "val": []}
    for epoch in range(epochs):
        peft_model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=fp16 and device.type == "cuda", dtype=torch.float16
            ):
                out = peft_model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss)
            if (step + 1) % log_every == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} loss={loss.item():.4f}")
        avg = running / max(len(train_loader), 1)
        history["train_loss"].append(avg)
        msg = f"[ColabPLM] epoch {epoch+1}/{epochs} train_loss={avg:.4f}"
        if val_loader is not None:
            m = evaluate(peft_model, val_loader, is_regression)
            history["val"].append(m)
            msg += " | val=" + ", ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(msg)

    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    bundle.tokenizer.save_pretrained(output_dir)
    print(f"[ColabPLM] adapter saved to {output_dir}")
    return peft_model, bundle, history


@torch.no_grad()
def predict_sequences(
    model, tokenizer, sequences: Iterable[str],
    is_t5: bool = False, max_length: int = 512,
):
    model.eval()
    device = next(model.parameters()).device
    seqs = [sanitize_sequence(s) for s in sequences]
    if is_t5:
        seqs = [" ".join(list(s)) for s in seqs]
    enc = tokenizer(
        seqs, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)
    logits = model(**enc).logits
    return logits.cpu()
