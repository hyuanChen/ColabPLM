"""Generate ColabPLM.ipynb programmatically so the notebook is reproducible."""
import json
from pathlib import Path

NB = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": []},
}, "cells": []}


def md(text): NB["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
def code(text): NB["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)})


md("""# ColabPLM — fine-tuning any PLM with LoRA inside Colab

This notebook adapts the **ColabSaprot** template into a model-agnostic
**ColabPLM**: a one-stop pipeline that lets you fine-tune any
HuggingFace-hosted Protein Language Model (ESM-2, ESM-C, ESM-3, ProtT5, SaProt…)
on a sequence-classification or regression task using parameter-efficient
LoRA adapters.

Sections:
1. Environment & GPU check
2. Install dependencies
3. Choose backbone & task
4. Data preparation (DeepLoc binary membrane vs soluble)
5. Build model + LoRA adapter
6. Train
7. Save and re-load the adapter
8. Inference demo
""")

md("## 1. Environment & GPU check")
code("""!nvidia-smi -L || echo 'no GPU — switch to GPU runtime via Runtime > Change runtime type'
import platform, sys
print('python', sys.version.split()[0], '|', platform.platform())""")

md("## 2. Install dependencies")
code("""%pip install -q transformers==4.46.3 peft==0.13.2 pytorch-lightning==2.4.0 \\
                  datasets scikit-learn pandas biopython safetensors""")

md("## 3. Choose a backbone PLM and downstream task\n\n"
   "Pick one of the entries below (or paste any HuggingFace seq-classification-capable "
   "PLM identifier). LoRA target modules are auto-detected per family.")
code("""MODEL_NAME = \"facebook/esm2_t12_35M_UR50D\"   # @param ['facebook/esm2_t6_8M_UR50D','facebook/esm2_t12_35M_UR50D','facebook/esm2_t30_150M_UR50D','facebook/esm2_t33_650M_UR50D','Rostlab/prot_t5_xl_uniref50','westlake-repl/SaProt_35M_AF2'] {allow-input: true}
TASK = \"deeploc_binary\"   # membrane (1) vs soluble (0)
NUM_LABELS  = 2
IS_REGRESSION = False
MAX_LENGTH = 256
RANK, ALPHA = 8, 16
EPOCHS, BATCH_SIZE, LR = 3, 16, 5e-4""")

md("## 4. Fetch the ColabPLM source modules\n\n"
   "We ship the ColabPLM library as plain `.py` files inline so the notebook is "
   "self-contained on Colab without needing a separate `git clone`.")
# Inline the four modules so the notebook is fully self-contained.
import_block = ""
for mod in ("loader.py", "data.py", "adapter.py", "trainer.py"):
    src = Path(__file__).resolve().parent.parent.joinpath("colab_plm", mod).read_text()
    import_block += f"# -- colab_plm/{mod} --\n{src}\n\n"

code(f"""import os, textwrap, pathlib
pathlib.Path('colab_plm').mkdir(exist_ok=True)
SRC = {json.dumps(import_block)}
for chunk in SRC.split('# -- colab_plm/'):
    if not chunk.strip():
        continue
    name, body = chunk.split(' --', 1)
    (pathlib.Path('colab_plm') / name).write_text(body.lstrip('\\n'))
pathlib.Path('colab_plm/__init__.py').write_text('''
from .loader import ColabPLMLoader
from .data import ProteinSequenceDataset, build_dataloaders, sanitize_sequence
from .adapter import inject_lora_adapter, suggest_target_modules
from .trainer import train_colab_plm, evaluate, predict_sequences
''')
print('wrote colab_plm/', os.listdir('colab_plm'))""")

md("## 5. Build the dataset (DeepLoc binary)")
code("""from datasets import load_dataset
import pandas as pd, os

os.makedirs('data', exist_ok=True)
ds = load_dataset('proteinea/deeploc')

def to_df(split, n):
    d = ds[split].to_pandas()
    d = d[d['membrane'].isin(['M','S'])]
    df = pd.DataFrame({'protein': d['input'].str[:1022], 'label': (d['membrane']=='M').astype(int)})
    return df.sample(n=min(n,len(df)), random_state=0).reset_index(drop=True)

train_df = to_df('train', 2000)
val_df   = to_df('test',  500)
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv',   index=False)
print('train', len(train_df), 'val', len(val_df))
train_df.head(3)""")

md("## 6. Train ESM-2 + LoRA")
code("""from colab_plm import train_colab_plm, predict_sequences

model, bundle, history = train_colab_plm(
    model_name=MODEL_NAME,
    train_csv='data/train.csv', val_csv='data/val.csv',
    num_labels=NUM_LABELS, output_dir='outputs/colab_plm_adapter',
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    rank=RANK, alpha=ALPHA, max_length=MAX_LENGTH,
    is_regression=IS_REGRESSION,
)
history""")

md("## 7. Reload the saved LoRA adapter (simulating a fresh runtime)")
code("""import importlib, colab_plm
importlib.reload(colab_plm)
from colab_plm import ColabPLMLoader
from peft import PeftModel

bundle = ColabPLMLoader.load_backbone(MODEL_NAME, num_labels=NUM_LABELS, is_regression=IS_REGRESSION)
reloaded = PeftModel.from_pretrained(bundle.model, 'outputs/colab_plm_adapter').eval().cuda()
print('reloaded — trainable params:')
reloaded.print_trainable_parameters()""")

md("## 8. Inference demo")
code("""demo = [
    'MKTIIALSYIFCLVFADYKDDDDK',                        # short signal peptide
    'MFLVLLALVAVSAQGNCKEHLTKLPELPSGCTKVTLQ' * 3,        # soluble-ish
    'FFLLAAVVVLLIIVVAAFFLLAVVI' * 6,                    # hydrophobic, membrane-like
]
logits = predict_sequences(reloaded, bundle.tokenizer, demo, is_t5=bundle.is_t5, max_length=MAX_LENGTH)
print('logits:'); print(logits)
print('predictions (0=soluble, 1=membrane):', logits.argmax(-1).tolist())""")

md("""## Notes
* Switching to ESM-C, ESM-3 or ProtT5 only requires changing ``MODEL_NAME``; the
  loader picks the correct tokenizer (with whitespace separation for T5) and
  the adapter builder auto-detects ``target_modules``.
* For regression tasks set ``IS_REGRESSION=True`` and ``NUM_LABELS=1``; the
  ``problem_type='regression'`` flag flows through automatically.
* The saved ``outputs/colab_plm_adapter`` directory contains only the LoRA
  matrices + the classification head (~1.6 MB for ESM-2-35M); the frozen
  backbone is fetched from HuggingFace on demand.""")

out = Path(__file__).resolve().parent / "ColabPLM.ipynb"
out.write_text(json.dumps(NB, indent=1))
print(f"wrote {out}")
