"""Command-line interface for FinProbTS-Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from finprobts.config import load_yaml_config
from finprobts.data import get_default_dataset_registry, handle_missing_values, price_to_log_return
from finprobts.evaluation import evaluate_forecasts
from finprobts.experiment import run_data_efficiency, run_experiment, run_synthetic_suite_benchmark
from finprobts.models import ForecastResult
from finprobts.synthetic import generate_synthetic_suite


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    config = load_yaml_config(args.config)
    dataset_config = dict(config.get("dataset", {}))
    dataset_name = dataset_config.pop("name")
    dataset = get_default_dataset_registry().load(dataset_name, **dataset_config)

    preprocessing = config.get("preprocessing", {})
    value_kind = preprocessing.get("value_kind", dataset.metadata.get("value_kind", "returns"))
    if preprocessing.get("price_to_log_return", False) or value_kind == "prices":
        dataset = price_to_log_return(dataset)
    dataset = handle_missing_values(dataset, method=preprocessing.get("missing_method", "ffill"))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        values=dataset.values,
        dates=dataset.dates.astype("datetime64[ns]").astype(str),
        asset_ids=np.asarray(dataset.asset_ids, dtype=str),
    )
    print(f"Prepared data saved to {output_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    result = run_experiment(args.config)
    print(f"Run complete: {result.output_dir}")
    print(json.dumps(result.forecast_metrics, indent=2, allow_nan=False))
    return 0


def _parse_csv_floats(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of floats.")
    return values


def _parse_csv_strings(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of names.")
    return values


def _cmd_data_efficiency(args: argparse.Namespace) -> int:
    result = run_data_efficiency(
        args.config,
        train_fractions=args.train_fractions,
        model_names=args.models,
        plot=True if args.plot else None,
    )
    total = len(result.rows)
    failures = sum(1 for row in result.rows if int(row.get("exit_code", 0)) != 0)
    print(f"Data-efficiency run complete: {result.output_dir}")
    print(f"Results CSV: {result.results_csv}")
    print(f"Results JSON: {result.results_json}")
    print(f"Completed settings: {total - failures}/{total}")
    if failures:
        print(f"Failed settings: {failures}; see logs under {result.output_dir / 'logs'}")
    if result.plot_paths:
        print("Plots:")
        for plot_path in result.plot_paths:
            print(f"  {plot_path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    if args.run_dir:
        forecast_path = Path(args.run_dir) / "forecast_samples.npz"
        output_path = Path(args.run_dir) / "forecast_metrics.json"
    else:
        forecast_path = Path(args.forecast_path)
        output_path = Path(args.output) if args.output else None

    result = ForecastResult.load_npz(str(forecast_path))
    metrics = evaluate_forecasts(result)
    print(json.dumps(metrics, indent=2, allow_nan=False))

    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)
    return 0


def _cmd_portfolio(args: argparse.Namespace) -> int:
    raise SystemExit("Portfolio backtesting is not implemented in the MVP.")


def _parse_levels(value: str) -> list[int]:
    levels = []
    for part in value.split(","):
        part = part.strip()
        if part:
            levels.append(int(part))
    if not levels:
        raise argparse.ArgumentTypeError("levels must contain at least one integer.")
    return levels


def _cmd_generate_synthetic(args: argparse.Namespace) -> int:
    manifest = generate_synthetic_suite(
        case=args.case,
        levels=args.levels,
        out_dir=args.out_dir,
        base_seed=args.base_seed,
        T=args.T,
        n_firms=args.n_firms,
        formats=args.formats.split(","),
    )
    print(f"Generated {len(manifest['datasets'])} synthetic dataset(s).")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


def _cmd_run_synthetic_suite(args: argparse.Namespace) -> int:
    result = run_synthetic_suite_benchmark(
        manifest_path=args.manifest,
        models=args.models,
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        model_config_dir=args.model_config_dir,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        train_size=args.train_size,
        val_size=args.val_size,
        num_samples=args.num_samples,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
        dry_run=args.dry_run,
        continue_on_error=not args.stop_on_error,
    )
    total = len(result.rows)
    failures = sum(1 for row in result.rows if int(row.get("exit_code", 0)) != 0)
    print(f"Synthetic-suite benchmark complete: {result.output_dir}")
    print(f"Resolved configs: {result.config_dir}")
    print(f"Completed settings: {total - failures}/{total}")
    if failures:
        print(f"Failed settings: {failures}")
    print("Result CSVs:")
    for model_name, csv_path in result.result_csvs.items():
        print(f"  {model_name}: {csv_path}")
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finprobts", description="FinProbTS-Bench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Load and preprocess a dataset.")
    prepare.add_argument("--config", required=True, help="Path to a YAML config.")
    prepare.add_argument("--output", default="outputs/prepared_data.npz", help="Output NPZ path.")
    prepare.set_defaults(func=_cmd_prepare_data)

    run = subparsers.add_parser("run", help="Run a forecasting experiment.")
    run.add_argument("--config", required=True, help="Path to a YAML config.")
    run.set_defaults(func=_cmd_run)

    data_efficiency = subparsers.add_parser("data-efficiency", help="Run train-data percentage sweeps.")
    data_efficiency.add_argument("--config", required=True, help="Path to a YAML config.")
    data_efficiency.add_argument(
        "--train-fractions",
        type=_parse_csv_floats,
        help="Optional comma-separated train fractions, e.g. 0.1,0.2,0.5,1.0.",
    )
    data_efficiency.add_argument(
        "--models",
        type=_parse_csv_strings,
        help="Optional comma-separated model names or model ids to run.",
    )
    data_efficiency.add_argument(
        "--plot",
        action="store_true",
        help="Generate CRPS data-efficiency plots after the sweep.",
    )
    data_efficiency.set_defaults(func=_cmd_data_efficiency)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate saved forecast samples.")
    evaluate.add_argument("--run-dir", help="Run directory containing forecast_samples.npz.")
    evaluate.add_argument("--forecast-path", help="Path to forecast_samples.npz.")
    evaluate.add_argument("--output", help="Optional output JSON path.")
    evaluate.set_defaults(func=_cmd_evaluate)

    portfolio = subparsers.add_parser("portfolio", help="Portfolio backtesting placeholder.")
    portfolio.set_defaults(func=_cmd_portfolio)

    synthetic = subparsers.add_parser("generate-synthetic", help="Generate notebook-style synthetic benchmark datasets.")
    synthetic.add_argument("--case", default="all", help="Case name or comma-separated names; use 'all' for every case.")
    synthetic.add_argument("--levels", type=_parse_levels, default=[1, 2, 3, 4, 5], help="Comma-separated levels, e.g. 1,2,3.")
    synthetic.add_argument("--out-dir", default="data/simulated", help="Output directory for generated datasets.")
    synthetic.add_argument("--base-seed", type=int, default=None, help="Base seed; final seed is base_seed + level.")
    synthetic.add_argument("--T", type=int, default=None, help="Override time-series length.")
    synthetic.add_argument("--n-firms", type=int, default=None, help="Override panel size when the case supports it.")
    synthetic.add_argument("--formats", default="csv", help="Comma-separated output formats: csv or csv,parquet.")
    synthetic.set_defaults(func=_cmd_generate_synthetic)

    synthetic_suite = subparsers.add_parser(
        "run-synthetic-suite",
        help="Run selected models over every generated synthetic dataset in a manifest.",
    )
    synthetic_suite.add_argument(
        "--manifest",
        default="data/simulated/manifest.json",
        help="Synthetic manifest written by generate-synthetic.",
    )
    synthetic_suite.add_argument(
        "--models",
        type=_parse_csv_strings,
        default=["deepvar"],
        help="Comma-separated models to run, e.g. deepvar,tempflow, or all.",
    )
    synthetic_suite.add_argument(
        "--output-dir",
        default="outputs/synthetic_suite",
        help="Directory for per-model result CSVs and experiment outputs.",
    )
    synthetic_suite.add_argument(
        "--config-dir",
        default="configs/generated/synthetic_suite",
        help="Directory where resolved per dataset/model YAML configs are saved.",
    )
    synthetic_suite.add_argument(
        "--model-config-dir",
        default="configs/model",
        help="Directory containing default model YAML snippets.",
    )
    synthetic_suite.add_argument("--context-length", type=int, default=96)
    synthetic_suite.add_argument("--prediction-length", type=int, default=1)
    synthetic_suite.add_argument("--stride", type=int, default=1)
    synthetic_suite.add_argument("--train-size", type=float, default=0.6)
    synthetic_suite.add_argument("--val-size", type=float, default=0.2)
    synthetic_suite.add_argument("--num-samples", type=int, default=100)
    synthetic_suite.add_argument("--max-epochs", type=int, default=None)
    synthetic_suite.add_argument("--batch-size", type=int, default=None)
    synthetic_suite.add_argument("--device", default=None, help="Override device for torch models, e.g. cpu or cuda.")
    synthetic_suite.add_argument(
        "--dry-run",
        action="store_true",
        help="Write resolved configs and summary CSVs without training models.",
    )
    synthetic_suite.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the suite at the first failed dataset/model run.",
    )
    synthetic_suite.set_defaults(func=_cmd_run_synthetic_suite)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "evaluate" and not (args.run_dir or args.forecast_path):
        parser.error("evaluate requires --run-dir or --forecast-path.")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
