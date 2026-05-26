"""End-to-end ColabPLM training driver."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab_plm import train_colab_plm, predict_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--train-csv", default="data/train.csv")
    ap.add_argument("--val-csv",   default="data/val.csv")
    ap.add_argument("--output",    default="outputs/esm2_lora")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--num-labels", type=int, default=2)
    ap.add_argument("--epochs",      type=int, default=3)
    ap.add_argument("--batch-size",  type=int, default=8)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--rank",        type=int, default=8)
    ap.add_argument("--alpha",       type=int, default=16)
    ap.add_argument("--max-length",  type=int, default=256)
    ap.add_argument("--regression",  action="store_true")
    ap.add_argument("--no-fp16",     action="store_true")
    args = ap.parse_args()

    model, bundle, history = train_colab_plm(
        model_name=args.model,
        train_csv=args.train_csv, val_csv=args.val_csv,
        num_labels=args.num_labels, output_dir=args.output,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        rank=args.rank, alpha=args.alpha, max_length=args.max_length,
        is_regression=args.regression, cache_dir=args.cache_dir,
        fp16=not args.no_fp16,
    )

    demo = [
        "MKTIIALSYIFCLVFADYKDDDDK",
        "FFLLAAVVVLLIIVVAAFFLLAVVI",
    ]
    logits = predict_sequences(model, bundle.tokenizer, demo, is_t5=bundle.is_t5,
                               max_length=args.max_length)
    print("[demo] logits:")
    print(logits)
    with open(os.path.join(args.output, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
