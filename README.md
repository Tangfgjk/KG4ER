# KG4ER ER Experiment Code

This repository contains the code-only version of the ER/KG4ER experiment pipeline.

Large datasets, generated KT exports, model checkpoints, experiment logs, run outputs, PDF files, and the local `ER/docs/` folder are intentionally not uploaded. Copy prepared data to the expected local paths before running experiments.

## Directory Layout

```text
ER/
  KG4ER/
    codes/          # training, testing, baselines, ablations, one-click runner
    data/           # data processing scripts only
    tests/          # regression tests for runner/baseline/KGE utilities
    requirements.txt
  docs-for-git/     # lightweight uploadable docs placeholder
```

## Data Placement

Put local data under:

```text
ER/KG4ER/data/Eedi/
ER/KG4ER/data/algebra2005/prepared_for_kt/
ER/KG4ER/data/assist2009/prepared_for_kt/
ER/KG4ER/data/statics2011/prepared_for_kt/
ER/KG4ER/data/XES3G5M-sub/prepared_for_kt/
```

The prepared folders should contain files such as `train.txt`, `test.txt`, `entities.dict`, `relations.dict`, `stu2know_mastery.json`, `stu2know_seq.json`, `stu2know_forget.json`, and `exercise_forget.json`.

## Install

```powershell
cd ER\KG4ER
pip install -r requirements.txt
```

Use a PyTorch/CUDA environment if available. The experiment runner defaults to CUDA when `--cuda auto` detects it.

## Run One Dataset

Run from `ER/KG4ER/codes`:

```powershell
cd ER\KG4ER\codes
python run_dataset_experiments.py `
  --dataset algebra2005 `
  --seeds 2024,2025,2026,2027,2028 `
  --cuda auto
```

Resume the latest run for the same dataset:

```powershell
python run_dataset_experiments.py `
  --dataset algebra2005 `
  --resume `
  --cuda auto
```

Run a single seed:

```powershell
python run_dataset_experiments.py `
  --dataset algebra2005 `
  --seeds 2024 `
  --cuda auto
```

Outputs are saved under `ER/KG4ER/runs/<dataset>/<dataset>_full_<timestamp>/`.

The one-click runner includes `ConvE_full`, ConvE ablations, `TransE`, `TransE-adv`, `RotatE`, `DistMult`, `ComplEx`, and the traditional baselines.

## Notes

- `ER/docs/` and PDF documents are kept local and are not part of this repository.
- `ER/docs-for-git/` is reserved for lightweight documents that are safe to upload later.
- Data files are not tracked. Copy prepared datasets separately on each new computer.
