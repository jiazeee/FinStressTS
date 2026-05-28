"""Data-efficiency experiments for probabilistic forecasting models."""

from __future__ import annotations

import copy
import csv
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from finprobts.config import load_yaml_config, save_yaml_config
from finprobts.data import (
    DatasetNormalizer,
    FinancialDataset,
    generate_rolling_windows,
    get_default_dataset_registry,
    time_train_val_test_split,
)
from finprobts.evaluation import evaluate_forecasts
from finprobts.experiment.runner import _inverse_result, _preprocess_dataset, _resolve_output_dir
from finprobts.models import get_default_model_registry


DEFAULT_TRAIN_FRACTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class DataEfficiencyResult:
    """Summary paths and rows produced by a data-efficiency run."""

    output_dir: Path
    results_csv: Path
    results_json: Path
    plot_paths: List[Path]
    rows: List[Dict[str, Any]]


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_clean(payload), handle, indent=2, allow_nan=False)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "model"


def _fraction_tag(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_fraction_values(raw: Any) -> List[float]:
    if raw is None:
        values = list(DEFAULT_TRAIN_FRACTIONS)
    elif isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]

    if not values:
        raise ValueError("At least one train fraction is required.")
    for value in values:
        if not 0.0 < value <= 1.0:
            raise ValueError("Each train fraction must be in (0, 1].")
    return values


def _normalize_model_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, str):
        return {"name": entry, "params": {}}
    if not isinstance(entry, dict):
        raise ValueError("Each model entry must be a model name or a mapping.")
    normalized = copy.deepcopy(entry)
    model_name = normalized.get("name") or normalized.get("type")
    if not model_name:
        raise ValueError("Each model entry requires a name.")
    params = dict(normalized.pop("params", {}))
    for key in list(normalized.keys()):
        if key not in {"id", "name", "type"}:
            params[key] = normalized.pop(key)
    normalized["name"] = str(model_name)
    normalized["params"] = params
    return normalized


def _configured_models(config: Dict[str, Any], override_model_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    efficiency_config = config.get("data_efficiency", {})
    raw_models = efficiency_config.get("models", config.get("models"))
    if raw_models is None:
        raw_models = config.get("model")
    if raw_models is None:
        raise ValueError("Config must include model, models, or data_efficiency.models.")
    if isinstance(raw_models, (str, dict)):
        raw_models = [raw_models]

    entries = [_normalize_model_entry(entry) for entry in raw_models]
    if not override_model_names:
        return entries

    requested = [name.strip() for name in override_model_names if name.strip()]
    by_name = {entry["name"]: entry for entry in entries}
    by_id = {str(entry.get("id")): entry for entry in entries if entry.get("id") is not None}
    selected = []
    for name in requested:
        selected.append(copy.deepcopy(by_id.get(name) or by_name.get(name) or {"name": name, "params": {}}))
    return selected


def _normalize_dataset_entry(entry: Any, index: int) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Each dataset entry must be a mapping.")
    dataset_config = copy.deepcopy(entry)
    dataset_id = dataset_config.pop("id", None)
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Each dataset entry requires a name.")
    dataset_path = dataset_config.get("path", "")
    if dataset_id is None:
        dataset_id = Path(str(dataset_path)).stem if dataset_path else f"{dataset_name}_{index + 1}"
    return {
        "id": str(dataset_id),
        "name": str(dataset_name),
        "path": str(dataset_path),
        "config": dataset_config,
    }


def _configured_datasets(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_datasets = config.get("datasets")
    if raw_datasets is None:
        dataset_config = config.get("dataset")
        if dataset_config is None:
            raise ValueError("Config must include dataset or datasets.")
        raw_datasets = [dataset_config]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("datasets must be a non-empty list.")
    return [_normalize_dataset_entry(entry, idx) for idx, entry in enumerate(raw_datasets)]


def _load_dataset_from_entry(dataset_entry: Dict[str, Any]) -> FinancialDataset:
    dataset_config = dict(dataset_entry["config"])
    dataset_name = dataset_config.pop("name", None)
    if not dataset_name:
        raise ValueError("dataset.name is required.")
    return get_default_dataset_registry().load(str(dataset_name), **dataset_config)


def _metric_fieldnames(quantiles: Iterable[float], coverage_levels: Iterable[float]) -> List[str]:
    fields = [
        "mae",
        "rmse",
        "mape",
        "nd",
        "crps",
        "crps_sum",
        "var_violation_rate",
        "expected_shortfall",
        "volatility_forecast_error",
        "correlation_forecast_error",
    ]
    fields.extend(f"quantile_loss_{float(q):.2f}" for q in quantiles)
    fields.extend(f"coverage_{float(level):.2f}" for level in coverage_levels)
    return fields


def _write_results_csv(rows: List[Dict[str, Any]], fieldnames: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({key for row in rows for key in row if key not in fieldnames})
    ordered = fieldnames + extra
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if _json_clean(row.get(name)) is None else _json_clean(row.get(name)) for name in ordered})


def _failure_row(
    *,
    dataset_entry: Dict[str, Any],
    model_entry: Dict[str, Any],
    train_fraction: float,
    output_dir: Path,
    exc: Exception,
    split_train: Optional[FinancialDataset] = None,
    split_val: Optional[FinancialDataset] = None,
    split_test: Optional[FinancialDataset] = None,
) -> Dict[str, Any]:
    model_name = str(model_entry.get("name", "unknown"))
    model_id = str(model_entry.get("id") or model_name)
    run_tag = (
        f"{_safe_name(dataset_entry['id'])}__"
        f"{_safe_name(model_id)}__train_fraction_{_fraction_tag(train_fraction)}"
    )
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_tag}.log"
    tb = traceback.format_exc()
    log_path.write_text(f"FAIL\nerror={repr(exc)}\n\n{tb}", encoding="utf-8")
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_entry["id"],
        "dataset_name": dataset_entry["name"],
        "dataset_path": dataset_entry["path"],
        "model": model_name,
        "model_id": model_id,
        "train_fraction": float(train_fraction),
        "full_train_timesteps": "" if split_train is None else split_train.num_timesteps,
        "val_timesteps": "" if split_val is None else split_val.num_timesteps,
        "test_timesteps": "" if split_test is None else split_test.num_timesteps,
        "num_assets": "" if split_train is None else split_train.num_assets,
        "exit_code": 1,
        "error": repr(exc),
        "log_file": str(log_path).replace("\\", "/"),
        "forecast_path": "",
    }


def _train_subset(dataset: FinancialDataset, fraction: float, right_aligned: bool) -> FinancialDataset:
    timesteps = int(np.ceil(dataset.num_timesteps * float(fraction)))
    timesteps = min(max(timesteps, 1), dataset.num_timesteps)
    if right_aligned:
        return dataset.slice_time(dataset.num_timesteps - timesteps, dataset.num_timesteps)
    return dataset.slice_time(0, timesteps)


def _maybe_windows(
    dataset: FinancialDataset,
    context_length: int,
    prediction_length: int,
    stride: int,
    metadata: Dict[str, Any],
):
    if dataset.num_timesteps < context_length + prediction_length:
        return None
    return generate_rolling_windows(dataset, context_length, prediction_length, stride, metadata=metadata)


def _make_model(model_entry: Dict[str, Any], run_seed: Any):
    params = dict(model_entry.get("params", {}))
    params.setdefault("seed", run_seed)
    return get_default_model_registry().create(str(model_entry["name"]), **params)


def _run_one_setting(
    *,
    config: Dict[str, Any],
    dataset_entry: Dict[str, Any],
    split_train: FinancialDataset,
    split_val: FinancialDataset,
    split_test: FinancialDataset,
    model_entry: Dict[str, Any],
    train_fraction: float,
    output_dir: Path,
) -> Dict[str, Any]:
    task_config = config.get("task", {})
    forecast_config = config.get("forecast", {})
    metrics_config = config.get("evaluation", {})
    efficiency_config = config.get("data_efficiency", {})

    context_length = int(task_config.get("context_length", 63))
    prediction_length = int(task_config.get("prediction_length", 1))
    stride = int(task_config.get("stride", 1))
    num_samples = int(forecast_config.get("num_samples", 100))
    standardize = bool(config.get("preprocessing", {}).get("standardize", True))
    right_aligned = bool(efficiency_config.get("right_aligned_train", True))
    save_forecasts = bool(efficiency_config.get("save_forecasts", False))

    train_raw = _train_subset(split_train, train_fraction, right_aligned=right_aligned)
    model_name = str(model_entry["name"])
    model_id = str(model_entry.get("id") or model_name)
    run_tag = (
        f"{_safe_name(dataset_entry['id'])}__"
        f"{_safe_name(model_id)}__train_fraction_{_fraction_tag(train_fraction)}"
    )
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_tag}.log"

    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_entry["id"],
        "dataset_name": dataset_entry["name"],
        "dataset_path": dataset_entry["path"],
        "model": model_name,
        "model_id": model_id,
        "train_fraction": float(train_fraction),
        "train_timesteps": train_raw.num_timesteps,
        "full_train_timesteps": split_train.num_timesteps,
        "val_timesteps": split_val.num_timesteps,
        "test_timesteps": split_test.num_timesteps,
        "num_assets": train_raw.num_assets,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "stride": stride,
        "num_samples": num_samples,
        "standardize": int(standardize),
        "right_aligned_train": int(right_aligned),
        "exit_code": 0,
        "error": "",
        "log_file": str(log_path).replace("\\", "/"),
        "forecast_path": "",
    }

    normalizer: Optional[DatasetNormalizer] = None
    if standardize:
        normalizer = DatasetNormalizer.fit(train_raw)
        train_data = normalizer.transform_dataset(train_raw)
        val_data = normalizer.transform_dataset(split_val)
        test_data = normalizer.transform_dataset(split_test)
    else:
        train_data = train_raw
        val_data = split_val
        test_data = split_test

    window_metadata = {
        "dataset_id": dataset_entry["id"],
        "train_fraction": float(train_fraction),
        "model_id": model_id,
    }
    train_windows = _maybe_windows(train_data, context_length, prediction_length, stride, window_metadata)
    val_windows = _maybe_windows(val_data, context_length, prediction_length, stride, window_metadata)
    test_windows = _maybe_windows(test_data, context_length, prediction_length, stride, window_metadata)
    if train_windows is None:
        raise ValueError(
            "Training slice is too short for the requested task: "
            f"train_timesteps={train_data.num_timesteps}, "
            f"context_length+prediction_length={context_length + prediction_length}."
        )
    if test_windows is None:
        raise ValueError(
            "Test split is too short for the requested task: "
            f"test_timesteps={test_data.num_timesteps}, "
            f"context_length+prediction_length={context_length + prediction_length}."
        )

    row["train_windows"] = len(train_windows)
    row["val_windows"] = 0 if val_windows is None else len(val_windows)
    row["test_windows"] = len(test_windows)

    model = _make_model(model_entry, config.get("run", {}).get("seed"))
    model.fit(train_windows, val_windows)
    forecast_result = model.predict(test_windows, num_samples=num_samples)
    eval_result = _inverse_result(forecast_result, normalizer)
    metrics = evaluate_forecasts(
        eval_result,
        quantiles=metrics_config.get("quantiles", (0.1, 0.5, 0.9)),
        coverage_levels=metrics_config.get("coverage_levels", (0.5, 0.9)),
        var_alpha=float(metrics_config.get("var_alpha", 0.05)),
    )
    row.update(metrics)

    if save_forecasts:
        forecast_path = (
            output_dir
            / "forecasts"
            / _safe_name(dataset_entry["id"])
            / _safe_name(model_id)
            / f"train_fraction_{_fraction_tag(train_fraction)}"
            / "forecast_samples.npz"
        )
        eval_result.save_npz(str(forecast_path))
        row["forecast_path"] = str(forecast_path).replace("\\", "/")

    log_path.write_text(
        "OK\n"
        f"dataset_id={dataset_entry['id']}\n"
        f"model={model_name}\n"
        f"train_fraction={train_fraction}\n"
        f"train_timesteps={train_raw.num_timesteps}\n"
        f"train_windows={len(train_windows)}\n"
        f"metrics={_json_clean(metrics)}\n",
        encoding="utf-8",
    )
    return row


def _should_plot_row(row: Dict[str, Any]) -> bool:
    try:
        exit_code = int(row.get("exit_code", 0))
        return (
            exit_code == 0
            and np.isfinite(float(row["crps"]))
            and np.isfinite(float(row["train_fraction"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _plot_crps_data_efficiency(rows: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    usable = [row for row in rows if _should_plot_row(row)]
    plot_paths: List[Path] = []
    if not usable:
        return plot_paths

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_ids = sorted({str(row["model_id"]) for row in usable})
    for model_id in model_ids:
        model_rows = [row for row in usable if str(row["model_id"]) == model_id]
        dataset_ids = sorted({str(row["dataset_id"]) for row in model_rows})
        if not dataset_ids:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        for dataset_id in dataset_ids:
            series = [row for row in model_rows if str(row["dataset_id"]) == dataset_id]
            series = sorted(series, key=lambda row: float(row["train_fraction"]))
            x_values = [float(row["train_fraction"]) * 100.0 for row in series]
            y_values = [float(row["crps"]) for row in series]
            ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=dataset_id)

        ax.set_title(f"Data Efficiency: {model_id}")
        ax.set_xlabel("% of training samples")
        ax.set_ylabel("CRPS")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        plot_path = plot_dir / f"data_efficiency_crps_{_safe_name(model_id)}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plot_paths.append(plot_path)
    return plot_paths


def run_data_efficiency(
    config_path: str,
    train_fractions: Optional[Sequence[float]] = None,
    model_names: Optional[Sequence[str]] = None,
    plot: Optional[bool] = None,
) -> DataEfficiencyResult:
    """Run data-efficiency experiments from a YAML config.

    Validation and test splits are fixed. For each requested percentage, the
    runner takes that fraction of the available training split, right-aligned by
    default so the reduced train set ends immediately before validation.
    """

    config = load_yaml_config(config_path)
    if "run" not in config:
        config["run"] = {"run_id": "data_efficiency"}
    elif not config["run"].get("run_id"):
        config["run"]["run_id"] = "data_efficiency"

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    efficiency_config = config.get("data_efficiency", {})
    fractions = _parse_fraction_values(
        train_fractions if train_fractions is not None else efficiency_config.get("train_fractions")
    )
    dataset_entries = _configured_datasets(config)
    model_entries = _configured_models(config, override_model_names=model_names)
    continue_on_error = bool(efficiency_config.get("continue_on_error", True))
    make_plots = bool(efficiency_config.get("plot", False)) if plot is None else bool(plot)

    metrics_config = config.get("evaluation", {})
    metric_fields = _metric_fieldnames(
        metrics_config.get("quantiles", (0.1, 0.5, 0.9)),
        metrics_config.get("coverage_levels", (0.5, 0.9)),
    )
    fieldnames = [
        "timestamp",
        "dataset_id",
        "dataset_name",
        "dataset_path",
        "model",
        "model_id",
        "train_fraction",
        "train_timesteps",
        "full_train_timesteps",
        "train_windows",
        "val_timesteps",
        "val_windows",
        "test_timesteps",
        "test_windows",
        "num_assets",
        "context_length",
        "prediction_length",
        "stride",
        "num_samples",
        "standardize",
        "right_aligned_train",
        *metric_fields,
        "exit_code",
        "error",
        "log_file",
        "forecast_path",
    ]

    rows: List[Dict[str, Any]] = []
    for dataset_entry in dataset_entries:
        split = None
        try:
            dataset = _preprocess_dataset(_load_dataset_from_entry(dataset_entry), config)
            split_config = config.get("split", {})
            split = time_train_val_test_split(
                dataset,
                train_size=float(split_config.get("train_size", 0.6)),
                val_size=float(split_config.get("val_size", 0.2)),
                test_size=split_config.get("test_size"),
            )
        except Exception as exc:
            for model_entry in model_entries:
                for fraction in fractions:
                    rows.append(
                        _failure_row(
                            dataset_entry=dataset_entry,
                            model_entry=model_entry,
                            train_fraction=fraction,
                            output_dir=output_dir,
                            exc=exc,
                        )
                    )
            if not continue_on_error:
                raise
            continue

        for model_entry in model_entries:
            for fraction in fractions:
                try:
                    row = _run_one_setting(
                        config=config,
                        dataset_entry=dataset_entry,
                        split_train=split.train,
                        split_val=split.val,
                        split_test=split.test,
                        model_entry=model_entry,
                        train_fraction=fraction,
                        output_dir=output_dir,
                    )
                except Exception as exc:
                    row = _failure_row(
                        dataset_entry=dataset_entry,
                        model_entry=model_entry,
                        train_fraction=fraction,
                        output_dir=output_dir,
                        exc=exc,
                        split_train=split.train,
                        split_val=split.val,
                        split_test=split.test,
                    )
                    if not continue_on_error:
                        rows.append(row)
                        raise
                rows.append(row)

    results_csv = output_dir / "data_efficiency_results.csv"
    results_json = output_dir / "data_efficiency_results.json"
    plot_paths = _plot_crps_data_efficiency(rows, output_dir) if make_plots else []
    _write_results_csv(rows, fieldnames, results_csv)
    _write_json({"rows": rows, "plot_paths": [str(path) for path in plot_paths]}, results_json)
    save_yaml_config(config, str(output_dir / "config.yaml"))

    return DataEfficiencyResult(
        output_dir=output_dir,
        results_csv=results_csv,
        results_json=results_json,
        plot_paths=plot_paths,
        rows=rows,
    )
