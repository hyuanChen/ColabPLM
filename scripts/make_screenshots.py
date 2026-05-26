"""Render visual artifacts proving the ColabPLM pipeline runs.

Generates four PNGs that double as the deliverable "screenshots":
  * training_curve.png      — train loss vs. val acc / f1 across epochs
  * confusion_matrix.png    — on the held-out validation split
  * training_log.png        — terminal-style render of the train run log
  * notebook_summary.png    — table of executed notebook cells
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colab_plm import ColabPLMLoader, build_dataloaders, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCR = ROOT / "screenshots"
SCR.mkdir(exist_ok=True)


def _curve():
    h = json.load(open(ROOT / "outputs/esm2_lora/history.json"))
    epochs = list(range(1, len(h["train_loss"]) + 1))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, h["train_loss"], "o-", label="train_loss", color="tab:red")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [v["acc"] for v in h["val"]], "s-", label="val_acc",
             color="tab:blue")
    ax2.plot(epochs, [v["f1"] for v in h["val"]],  "^-", label="val_f1",
             color="tab:green")
    ax2.set_ylabel("metric"); ax2.set_ylim(0.5, 1.0)
    fig.suptitle("ColabPLM — ESM-2 (35M) + LoRA on DeepLoc binary")
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.92))
    fig.tight_layout()
    fig.savefig(SCR / "training_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote training_curve.png")


def _confusion():
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    bundle = ColabPLMLoader.load_backbone(
        "facebook/esm2_t12_35M_UR50D", num_labels=2,
        cache_dir=str(ROOT / "hf_cache"),
    )
    model = PeftModel.from_pretrained(bundle.model,
                                      str(ROOT / "outputs/esm2_lora")).eval().cuda()
    _, val_loader = build_dataloaders(
        bundle.tokenizer, str(ROOT / "data/train.csv"),
        str(ROOT / "data/val.csv"), batch_size=16, max_length=256,
    )
    preds, golds = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())
            golds.extend(batch["labels"].cpu().tolist())
    cm = confusion_matrix(golds, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["soluble (0)", "membrane (1)"])
    ax.set_yticklabels(["soluble (0)", "membrane (1)"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Validation confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(SCR / "confusion_matrix.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote confusion_matrix.png")
    return cm


def _log_image():
    log = (ROOT / "outputs/train_log.txt").read_text()
    lines = log.splitlines()
    head = lines[:18] + ["..."] + lines[-12:]
    text = "\n".join(head)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axis("off")
    ax.set_facecolor("#0b1020")
    fig.patch.set_facecolor("#0b1020")
    ax.text(0.01, 0.99, text, family="monospace", fontsize=9.5,
            color="#d6ffd6", va="top", ha="left")
    ax.set_title("scripts/run_train.py — terminal output",
                 color="white", loc="left")
    fig.savefig(SCR / "training_log.png", dpi=140, bbox_inches="tight",
                facecolor="#0b1020")
    plt.close(fig)
    print("wrote training_log.png")


def _notebook_summary():
    nb = json.loads((ROOT / "notebooks/ColabPLM.executed.ipynb").read_text())
    rows = []
    for i, c in enumerate(nb["cells"]):
        kind = c["cell_type"]
        src = "".join(c.get("source", []))[:60].replace("\n", " ↵ ")
        err = any(o.get("output_type") == "error" for o in c.get("outputs", []))
        rows.append((i, kind, "FAIL" if err else "OK", src))
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 1.0))
    ax.axis("off")
    table = ax.table(
        cellText=[[r[0], r[1], r[2], r[3]] for r in rows],
        colLabels=["#", "type", "status", "source (first 60 chars)"],
        loc="upper left", cellLoc="left", colWidths=[0.05, 0.12, 0.10, 0.73],
    )
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.2)
    for r in range(1, len(rows) + 1):
        col = "#d4f7d4" if rows[r - 1][2] == "OK" else "#ffd4d4"
        for c in range(4):
            table[(r, c)].set_facecolor(col)
    ax.set_title("notebooks/ColabPLM.executed.ipynb — cell-by-cell status",
                 loc="left")
    fig.savefig(SCR / "notebook_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote notebook_summary.png")


if __name__ == "__main__":
    _curve()
    _confusion()
    _log_image()
    _notebook_summary()
