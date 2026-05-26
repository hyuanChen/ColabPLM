# ColabPLM

A model-agnostic refactor of the **ColabSaprot** template that lets you LoRA
fine-tune any HuggingFace-hosted PLM (ESM-2, ESM-C, ESM-3, ProtT5, SaProt …)
inside Google Colab.

```
ColabPLM/
├── colab_plm/            # library (5 modules)
│   ├── loader.py         # family-aware tokenizer + model loader
│   ├── data.py           # CSV → tokenised tensors (sanitise, T5 spacing)
│   ├── adapter.py        # auto-detected LoRA target_modules + classifier save
│   └── trainer.py        # minimal AMP training / eval / inference
├── scripts/
│   ├── prepare_data.py   # DeepLoc binary → data/train.csv, data/val.csv
│   ├── run_train.py      # one-shot training driver
│   └── make_screenshots.py
├── notebooks/
│   ├── ColabPLM.ipynb            # clean notebook
│   ├── ColabPLM.executed.ipynb   # executed copy with outputs (proof)
│   └── ColabPLM.executed.html
├── data/                 # generated DeepLoc CSVs
├── hf_cache/             # local HuggingFace cache
├── outputs/esm2_lora/    # saved adapter (LoRA + classifier ≈ 1.6 MB)
├── screenshots/          # 4 PNGs proving the pipeline runs
└── report/
    ├── TECHNICAL_REPORT.md
    ├── TECHNICAL_REPORT.html
    └── TECHNICAL_REPORT.pdf
```

## Local reproduction (RTX 4090 / Linux)

```bash
conda activate pretrain
pip install transformers==4.46.3 peft==0.13.2 pytorch-lightning==2.4.0 \
            datasets scikit-learn pandas biopython

export HF_HOME=$PWD/hf_cache
https_proxy=http://127.0.0.1:7893 http_proxy=http://127.0.0.1:7893 \
    python scripts/prepare_data.py --out data
python scripts/run_train.py --epochs 3 --batch-size 16
python scripts/make_screenshots.py
```

Expected: val accuracy ≈ 0.86, macro-F1 ≈ 0.85 on the 500-sample DeepLoc test
slice in < 2 min wall-clock.

## Colab link
Push this directory to GitHub, then open:

```
https://colab.research.google.com/github/hyuanChen/ColabPLM/blob/main/notebooks/ColabPLM.ipynb
```

The notebook inlines `colab_plm/*.py`, so it works on a clean Colab runtime
without any `git clone`.
