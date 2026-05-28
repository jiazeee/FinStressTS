# FinStressTS Paper Code

This repository is the clean, code-only paper release for **FinStressTS**. It keeps the Python package and command-line interface named `finprobts` so existing experiment configs and imports remain compatible, while the repository itself is branded as the FinStressTS paper-code snapshot.

The broader FinProbTS-Bench project can keep evolving as a general benchmark. This repository is intended to be the frozen, citable implementation used for the accepted paper, with small examples and reproducibility instructions instead of private data, generated datasets, or large outputs.

## What Is Included

- Canonical financial time-series data loading for wide and long CSV/Parquet files.
- Preprocessing utilities for log returns, missing values, rolling normalization, chronological splits, and rolling windows.
- Unified probabilistic forecasting interface with `ForecastResult` samples shaped `[num_windows, num_samples, prediction_length, num_assets]`.
- Models currently exposed through the benchmark interface: `naive`, `deepar`, `deepvar`, `tempflow`, `timegrad`, `timemcl`, `ratd`, and `tsflow`.
- Forecast evaluation metrics including point metrics, quantile loss, empirical coverage, CRPS approximations, and finance-oriented forecast diagnostics.
- Synthetic data generation utilities and a data-efficiency runner with optional CRPS plots.
- Tests and minimal public example data under `data/example/`.

The deep model modules are native benchmark implementations adapted from paper and upstream architecture references. Upstream repositories were used as references; their implementations are not vendored wholesale. See each model's `REFERENCE.md` for paper, upstream repository, license, and deviations.

## Installation

```powershell
cd "C:\Users\Sun Jiaze\PycharmProjects\FinStressTS"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

For deep models, install the torch extra:

```powershell
python -m pip install -e .[dev,torch]
```

## Quickstart

Run the lightweight example using the bundled public toy returns dataset:

```powershell
finprobts run --config configs/example_crypto_naive.yaml
```

Generate synthetic benchmark datasets:

```powershell
finprobts generate-synthetic --case all --levels 1,2,3,4,5 --out-dir data/simulated --base-seed 123 --T 20000 --n-firms 50 --formats csv
```

Run a data-efficiency sweep with plots:

```powershell
finprobts data-efficiency --config configs/example_data_efficiency_naive.yaml --plot
```

Outputs are written under `outputs/` by default and are intentionally ignored by git.

## Synthetic Dataset Parameters

The default synthetic parameters are defined in `finprobts/synthetic/presets.py`. Each synthetic case has:

- a `fixed` block for parameters shared across all five levels, such as `T`, `n_firms`, `n_factors`, burn-in, baseline volatilities, and common simulator constants;
- a `levels` list for the difficulty-specific parameters for levels 1-5;
- a `default_base_seed`, where the actual generated seed is `default_base_seed + level` unless `--base-seed` is provided.

At generation time, `finprobts/synthetic/generator.py` calls `get_case_config(case, level)`, merges `fixed` with the selected level config, applies CLI overrides such as `--T`, `--n-firms`, and `--base-seed`, and then passes the resolved config into the matching simulator class under `finprobts/simulators/`.

Users who only want to change benchmark scale should use CLI flags:

```powershell
finprobts generate-synthetic --case case1_garch --levels 1,2 --T 5000 --n-firms 20 --base-seed 777
```

Users who want to change the actual data-generating process should edit or copy `finprobts/synthetic/presets.py`, for example by changing the GARCH persistence parameters, heavy-tail degrees of freedom, Hawkes jump intensity, regime transition settings, or ZIP jump probabilities. Each generated dataset also writes a `.meta.json` file containing the resolved config used for that run.

## Synthetic Suite Benchmarks

After generating the full 30-dataset synthetic suite, run selected models over every dataset from the manifest:

```powershell
finprobts run-synthetic-suite --models deepvar
```

This command reads `data/simulated/manifest.json`, writes one resolved YAML per dataset/model under `configs/generated/synthetic_suite/<model>/`, runs the usual preprocessing, split, rolling-window, training, forecasting, and evaluation pipeline, then writes one result table per model:

```text
outputs/synthetic_suite/results_deepvar.csv
outputs/synthetic_suite/results_deepvar.json
outputs/synthetic_suite/synthetic_suite_summary.json
```

Select several models with a comma-separated list:

```powershell
finprobts run-synthetic-suite --models naive,deepvar,tempflow
```

Use `--models all` to run every registered model. For a quick smoke test before a long run, generate configs and summary shells without training:

```powershell
finprobts run-synthetic-suite --models deepvar --dry-run
```

Common runtime overrides are available:

```powershell
finprobts run-synthetic-suite --models deepvar --max-epochs 1 --num-samples 20 --device cpu
```

For open-source release, the recommended default is to publish the simulator presets, commands, and batch runner rather than committing all generated data and output artifacts. Generated datasets and outputs are ignored by git.

## Data Format

Wide format expects one date/time column and one target column per asset:

```text
date,asset_a,asset_b,asset_c
2020-01-01,0.001,0.0004,-0.0002
```

Long format expects one row per time and asset:

```text
time,series_id,y
0,asset_a,0.001
0,asset_b,0.0004
```

The canonical in-memory representation aligns assets into a `[time, asset]` target matrix before producing rolling windows.

## Benchmark Contract

All models receive the same canonical `RollingWindowDataset` from the runner. Adapters translate that canonical input into model-specific tensors internally. All models return `ForecastResult`, whose sample tensor has shape `[num_windows, num_samples, prediction_length, num_assets]`. This lets the same evaluator compare marginal and joint probabilistic models through one output contract.

The default paper-style forecasting setup is one-step ahead forecasting with `prediction_length: 1` and `context_length: 96`.

## Reproducibility

See `REPRODUCIBILITY.md` for the install, smoke-test, synthetic generation, and paper-output workflow. Fill in final paper metadata in `CITATION.cff` before public release.

## License

MIT. See `LICENSE`.
