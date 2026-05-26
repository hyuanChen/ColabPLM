"""Build train/val CSVs for the DeepLoc binary (membrane vs soluble) task.

Order of preference:
  1. Hugging Face dataset ``proteinea/deeploc_binary`` (requires network).
  2. A bundled synthetic fallback that lets the pipeline run with zero
     downloads — useful for unit-testing the LoRA + training loop offline.

Run with the project's proxy if you want the real dataset, e.g.::

    https_proxy=http://127.0.0.1:7894 http_proxy=http://127.0.0.1:7894 \\
        python scripts/prepare_data.py --out data/
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import pandas as pd

_HYDROPHOBIC = list("AVLIMFW")
_HYDROPHILIC = list("DEKRSNQT")


def _synthetic_split(
    n_per_class: int = 200, length: tuple = (80, 220), seed: int = 0
) -> pd.DataFrame:
    """Generate a class-separable toy dataset so we can verify learning."""
    rng = random.Random(seed)
    rows: List[Tuple[str, int]] = []
    for label, palette, fill in (
        (1, _HYDROPHOBIC, "GAVLI"),   # "membrane-like" — hydrophobic-rich
        (0, _HYDROPHILIC, "GSTNQ"),   # "soluble-like"  — polar-rich
    ):
        for _ in range(n_per_class):
            L = rng.randint(*length)
            seq = "".join(
                rng.choice(palette) if rng.random() < 0.65 else rng.choice(fill)
                for _ in range(L)
            )
            rows.append((seq, label))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["protein", "label"])


def _try_huggingface(
    name: str = "proteinea/deeploc",
    cache_dir: str | None = None,
    max_train: int | None = 2000,
    max_val: int | None = 500,
    max_len: int = 1022,
):
    """Download DeepLoc and convert to ``protein``/``label`` schema.

    The ``proteinea/deeploc`` dataset has columns: ``input``, ``loc``,
    ``membrane`` (``M`` for membrane-bound, ``S`` for soluble).  We use the
    binary ``membrane`` flag as the classification target.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[prepare_data] datasets not installed: {e}")
        return None
    try:
        ds = load_dataset(name, cache_dir=cache_dir)
    except Exception as e:
        print(f"[prepare_data] could not download {name}: {e}")
        return None

    def _to_df(split):
        d = ds[split].to_pandas()
        if "membrane" in d.columns and "input" in d.columns:
            d = d[d["membrane"].isin(["M", "S"])]
            out = pd.DataFrame({
                "protein": d["input"].str[:max_len].values,
                "label":   (d["membrane"] == "M").astype(int).values,
            })
        else:
            seq_col = "sequence" if "sequence" in d.columns else "protein"
            lbl_col = "label" if "label" in d.columns else "labels"
            out = d[[seq_col, lbl_col]].rename(
                columns={seq_col: "protein", lbl_col: "label"}
            )
        return out.reset_index(drop=True)

    frames = {s: _to_df(s) for s in ds.keys()}
    if "train" in frames and "test" in frames:
        train, val = frames["train"], frames["test"]
    else:
        only = next(iter(frames.values()))
        cut = int(len(only) * 0.8)
        train, val = only.iloc[:cut].reset_index(drop=True), only.iloc[cut:].reset_index(drop=True)
    if max_train:
        train = train.sample(n=min(max_train, len(train)), random_state=0).reset_index(drop=True)
    if max_val:
        val = val.sample(n=min(max_val, len(val)), random_state=0).reset_index(drop=True)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--max-val",   type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_df = val_df = None
    if not args.synthetic_only:
        got = _try_huggingface(cache_dir=args.cache_dir,
                               max_train=args.max_train, max_val=args.max_val)
        if isinstance(got, tuple):
            train_df, val_df = got
            print(f"[prepare_data] DeepLoc downloaded — train={len(train_df)} val={len(val_df)}")

    if train_df is None:
        print("[prepare_data] falling back to synthetic dataset")
        df = _synthetic_split(n_per_class=args.n_per_class)
        cut = int(0.8 * len(df))
        train_df, val_df = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

    train_path = os.path.join(args.out, "train.csv")
    val_path = os.path.join(args.out, "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"[prepare_data] wrote {train_path} ({len(train_df)}), {val_path} ({len(val_df)})")
    print(train_df.head(3).to_string())


if __name__ == "__main__":
    main()
