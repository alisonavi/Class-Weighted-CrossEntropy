# Midterm project, Group 9 topic

**Java vulnerability detection with CodeBERT and class-weighted cross-entropy**
MSITM 5134 Management of Organizational Data, KBTU, autumn 2026.
Reference study: Shestov et al., "Finetuning Large Language Models for Vulnerability Detection",
IEEE Access 13 (2025), doi:10.1109/ACCESS.2025.3546700.

Bekzat Saulebay (25MD0294), individual work on the Group 9 topic.

The task, the method and every result are explained in `report/Midterm_Report.pdf`.
This file only says how to run the code.

## What is in the folder

| Path | What it is |
| --- | --- |
| `config.json` | model configuration: data commit and checksums, model revision, hyperparameters, all runs |
| `common.py` | paths, logging, warehouse connection, chart style |
| `data_pipeline.py` | acquisition, cleaning, quality checks, load into the SQLite warehouse |
| `train.py` | fine-tunes CodeBERT with class-weighted cross-entropy (primary method and ablations) |
| `alternative.py` | alternative method: frozen CodeBERT embeddings + class-weighted logistic regression |
| `baselines.py` | reference models: majority class, code length only, TF-IDF + logistic regression |
| `evaluate.py` | test metrics, bootstrap intervals, charts; writes results back into the warehouse |
| `demo.py` | live demo: checks the model hash and scores Java functions |
| `midterm_project.ipynb` | the whole project step by step, already executed |
| `runs/` | training logs (`train.log`, `train_log.csv`), `run_summary.json` and predictions for every run |
| `out/` | quality tables, metrics, `metrics.json`, all figures |
| `diagrams/` | data lifecycle, system architecture and ER diagram (SVG and PNG) |
| `report/` | technical report (PDF) and its HTML source |
| `presentation/` | slides for the defence |

Not in the repository (see `.gitignore`): `checkpoints/` (fine-tuned weights, about 500 MB per run) and
`data/`. The dataset has no licence, so it is not redistributed: `data_pipeline.py` downloads it from the
pinned commit and rebuilds the warehouse.

## Environment

Python 3.12. I used an Apple M3 Pro (36 GB) with the MPS GPU. The code also runs on CUDA or on
the CPU, but CPU training is very slow.

```bash
python3.12 -m venv ~/venvs/mood-midterm          # keep the venv outside this folder
~/venvs/mood-midterm/bin/pip install -r requirements.txt
```

## Run everything

```bash
PY=~/venvs/mood-midterm/bin/python
$PY data_pipeline.py                                   # under a minute
$PY train.py --run codebert_weighted                   # primary method, about 50 min on M3 Pro
$PY train.py --run codebert_plain                      # same model without class weights, about 50 min
$PY train.py --run codebert_sqrt                       # milder weights, about 50 min
$PY train.py --run codebert_weighted_balanced_set      # the balanced dataset, about 15 min
$PY alternative.py                                     # alternative method, about 3 min
$PY baselines.py                                       # under a minute
$PY evaluate.py                                        # under a minute
$PY demo.py --sample 2                                 # live demo
```

`data_pipeline.py` downloads the two dataset archives from GitHub at a fixed commit and the MITRE
CWE 4.20 catalog, and stops if a SHA-256 does not match `config.json`. The first `train.py` run
downloads CodeBERT at a fixed revision from Hugging Face. After that nothing goes to the network:
models load from the local cache and the background safetensors conversion in `transformers` is
switched off in `common.py`.

To try the model on your own code, put one Java method in a file and run
`$PY demo.py path/to/Method.java`.

## Reproducibility

Seeds are fixed (42) and the data steps are exact: the same commit gives the same rows, the same
quality issues and the same splits on every run. GPU training on MPS is not bit-for-bit
deterministic, so a new training run gives slightly different scores. The numbers in the report
come from the runs saved in `runs/`; `evaluate.py` recomputes every metric from those saved
predictions, and gives the same result each time.

## Data and licence

Dataset: <https://github.com/rmusab/vul-llm-finetune/tree/main/Datasets>, commit `862ed17`.
The repository had no licence file at that commit, so I do not redistribute the code samples;
the pipeline downloads them. The labels come from vulnerability-fixing commits (CVEfixes, a
manually curated dataset, VCMatch) and were not checked by a person.
