"""Backbone loader for ColabPLM.

A unified factory that hides the per-PLM quirks (ProtT5 needs spaced tokens,
ESM-2 accepts raw strings, ESM-C/ESM3 may need trust_remote_code, etc.) so the
rest of the pipeline can stay model-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class BackboneBundle:
    tokenizer: object
    model: torch.nn.Module
    is_t5: bool
    family: str   # "esm2" | "esmc" | "esm3" | "prott5" | "saprot" | "generic"
    hidden_size: int


def _detect_family(model_name: str) -> str:
    n = model_name.lower()
    if "saprot" in n:
        return "saprot"
    if "esm3" in n or "esm-3" in n:
        return "esm3"
    if "esmc" in n or "esm-c" in n or "esm_c" in n:
        return "esmc"
    if "esm2" in n or "esm-2" in n or "esm_2" in n or "esm" in n:
        return "esm2"
    if "prot_t5" in n or "prott5" in n or "t5" in n:
        return "prott5"
    return "generic"


class ColabPLMLoader:
    """Load any seq-classification-capable PLM from a HF identifier or local path."""

    @staticmethod
    def load_backbone(
        model_name_or_path: str,
        num_labels: int,
        is_regression: bool = False,
        cache_dir: str | None = None,
    ) -> BackboneBundle:
        family = _detect_family(model_name_or_path)
        is_t5 = family == "prott5"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            do_lower_case=False,
        )

        problem_type = "regression" if is_regression else "single_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type=problem_type,
            trust_remote_code=True,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )

        hidden_size = getattr(model.config, "hidden_size", None) or getattr(
            model.config, "d_model", -1
        )

        return BackboneBundle(
            tokenizer=tokenizer,
            model=model,
            is_t5=is_t5,
            family=family,
            hidden_size=hidden_size,
        )
