"""LoRA injection.

ColabSaprot's adapter step assumes SaProt's attention naming.  Different PLM
families name their projection matrices differently, so we discover the right
``target_modules`` from the model itself rather than hard-coding.
"""
from __future__ import annotations

from typing import Iterable, List

import torch
from peft import LoraConfig, TaskType, get_peft_model

# Ordered by preference — first family of patterns that exists in the model wins.
_PATTERN_GROUPS: List[List[str]] = [
    ["q_proj", "v_proj"],                # ESM-C / ESM-3 / LLaMA-style
    ["query", "value"],                  # ESM-2 / BERT / SaProt
    ["SelfAttention.q", "SelfAttention.v"],  # ProtT5 / T5
]


_HEAD_PATTERNS = ["classifier", "score", "classification_head"]


def suggest_modules_to_save(model: torch.nn.Module) -> List[str]:
    """Identify the freshly-initialised classification head so PEFT persists it."""
    names = {n for n, _ in model.named_modules()}
    out: list[str] = []
    for p in _HEAD_PATTERNS:
        if any(n == p or n.endswith("." + p) for n in names):
            out.append(p)
    return out


def suggest_target_modules(model: torch.nn.Module) -> List[str]:
    names = {n for n, _ in model.named_modules()}

    def _present(pat: str) -> bool:
        return any(n.endswith(pat) or n.endswith("." + pat) for n in names)

    for group in _PATTERN_GROUPS:
        if all(_present(p) for p in group):
            return group
    raise RuntimeError(
        "Could not auto-detect LoRA target modules.  Pass target_modules explicitly.  "
        "Sampled module names: " + ", ".join(list(names)[:10])
    )


def inject_lora_adapter(
    model: torch.nn.Module,
    target_modules: Iterable[str] | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    task_type: TaskType = TaskType.SEQ_CLS,
):
    if target_modules is None:
        target_modules = suggest_target_modules(model)
    modules_to_save = suggest_modules_to_save(model) or None
    cfg = LoraConfig(
        task_type=task_type,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        modules_to_save=modules_to_save,
        bias="none",
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model
