# Reproducibility

This paper-code repository is designed to be runnable without private data. The bundled dataset under `data/example/` is intentionally tiny and public; use synthetic generation for larger reproducible experiments.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Install deep-learning support when running neural models:

```powershell
python -m pip install -e .[dev,torch]
```

## Smoke Checks

```powershell
python -m compileall finprobts tests
python -m pytest
python -m finprobts.cli --help
```

## Generate Synthetic Data

```powershell
finprobts generate-synthetic --case all --levels 1,2,3,4,5 --out-dir data/simulated --base-seed 123 --T 20000 --n-firms 50 --formats csv
```

Generated datasets, forecasts, checkpoints, and plots are ignored by git.

## Run Experiments

Start with the lightweight baseline:

```powershell
finprobts run --config configs/example_crypto_naive.yaml
```

Run model-specific configs by swapping the config file, for example:

```powershell
finprobts run --config configs/example_crypto_deepvar.yaml
```

For paper tables and figures, record the exact config files, random seeds, git commit hash, Python version, and package versions used in each run.

## Paper Metadata TODO

Before release, update `CITATION.cff` with the final author list, title, venue, year, DOI, and repository URL.
