"""Batch runner for generated synthetic benchmark suites."""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from finprobts.config import load_yaml_config, save_yaml_config
from finprobts.experiment.runner import run_experiment
from finprobts.models import get_default_model_registry


DEFAULT_DATASET_CONFIG = {
    "name": "custom_csv",
    "format": "long",
    "date_column": "time",
    "asset_id_column": "series_id",
    "target_column": "y",
    "feature_columns": None,
    "time_index": "relative",
    "freq": "D",
    "validate_regular": True,
}


@dataclass
class SyntheticSuiteBenchmarkResult:
    """Paths and summary rows produced by a synthetic-suite benchmark."""

    output_dir: Path
    config_dir: Path
    result_csvs: Dict[str, Path]
    result_jsons: Dict[str, Path]
    rows: List[Dict[str, Any]]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default, allow_nan=False)


def _write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_json_default, allow_nan=False)
    return value


def _safe_model_token(name: str) -> str:
    return str(name).strip().lower().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_manifest(path: str) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Synthetic manifest not found: {manifest_path}. "
            "Run `finprobts generate-synthetic` first."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Manifest {manifest_path} does not contain a non-empty datasets list.")
    return manifest


def _resolve_models(models: Optional[Iterable[str]]) -> List[str]:
    registry = get_default_model_registry()
    available = registry.names()
    if models is None:
        return ["deepvar"]

    resolved: List[str] = []
    for item in models:
        for part in str(item).split(","):
            model = part.strip().lower()
            if not model:
                continue
            if model == "all":
                resolved.extend(available)
            else:
                resolved.append(model)

    resolved = list(dict.fromkeys(resolved))
    unknown = sorted(set(resolved) - set(available))
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}. Available models: {available}")
    if not resolved:
        raise ValueError("At least one model must be selected.")
    return resolved


def _load_model_template(model_name: str, model_config_dir: Optional[str]) -> Dict[str, Any]:
    if model_config_dir:
        path = Path(model_config_dir) / f"{model_name}.yaml"
        if path.exists():
            template = load_yaml_config(str(path))
            model = dict(template.get("model", {}))
            model.setdefault("name", model_name)
            return {"model": model, "forecast": dict(template.get("forecast", {}))}
    return {"model": {"name": model_name, "params": {}}, "forecast": {}}


def _with_model_overrides(
    model_name: str,
    model_template: Mapping[str, Any],
    num_samples: Optional[int],
    max_epochs: Optional[int],
    batch_size: Optional[int],
    device: Optional[str],
) -> Dict[str, Any]:
    model = dict(model_template.get("model", {}))
    model["name"] = model.get("name") or model_name
    params = dict(model.get("params", {}))
    if model_name != "naive":
        if max_epochs is not None:
            params["max_epochs"] = int(max_epochs)
        if batch_size is not None:
            params["batch_size"] = int(batch_size)
        if device is not None:
            params["device"] = str(device)
    model["params"] = params

    forecast = dict(model_template.get("forecast", {}))
    if num_samples is not None:
        forecast["num_samples"] = int(num_samples)
    forecast.setdefault("num_samples", 100)
    return {"model": model, "forecast": forecast}


def _dataset_csv_path(dataset_entry: Mapping[str, Any]) -> str:
    csv_path = dataset_entry.get("csv_path")
    if not csv_path:
        raise ValueError(f"Manifest entry lacks csv_path: {dataset_entry}")
    return str(csv_path)


def _dataset_id(dataset_entry: Mapping[str, Any]) -> str:
    return str(dataset_entry.get("tag") or f"{dataset_entry.get('case')}_level{dataset_entry.get('level')}")


def build_synthetic_suite_config(
    dataset_entry: Mapping[str, Any],
    model_name: str,
    model_template: Mapping[str, Any],
    output_dir: str,
    context_length: int = 96,
    prediction_length: int = 1,
    stride: int = 1,
    train_size: float = 0.6,
    val_size: float = 0.2,
    num_samples: Optional[int] = 100,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a resolved single-run config for one synthetic dataset/model pair."""

    dataset_id = _dataset_id(dataset_entry)
    model_token = _safe_model_token(model_name)
    overrides = _with_model_overrides(
        model_name=model_token,
        model_template=model_template,
        num_samples=num_samples,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
    )
    dataset_config = dict(DEFAULT_DATASET_CONFIG)
    dataset_config["path"] = _dataset_csv_path(dataset_entry)

    return {
        "run": {
            "run_id": f"{dataset_id}_{model_token}",
            "output_dir": str(Path(output_dir) / model_token),
            "seed": int(dataset_entry.get("seed", 42)),
        },
        "dataset": dataset_config,
        "preprocessing": {
            "value_kind": "returns",
            "price_to_log_return": False,
            "missing_method": "ffill",
            "standardize": True,
        },
        "split": {
            "train_size": float(train_size),
            "val_size": float(val_size),
        },
        "task": {
            "context_length": int(context_length),
            "prediction_length": int(prediction_length),
            "stride": int(stride),
        },
        "model": overrides["model"],
        "forecast": overrides["forecast"],
        "evaluation": {
            "quantiles": [0.1, 0.5, 0.9],
            "coverage_levels": [0.5, 0.9],
            "var_alpha": 0.05,
        },
    }


def _base_row(
    dataset_entry: Mapping[str, Any],
    model_name: str,
    config_path: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    summary = dict(dataset_entry.get("summary", {}) or {})
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": _dataset_id(dataset_entry),
        "case": dataset_entry.get("case"),
        "level": dataset_entry.get("level"),
        "seed": dataset_entry.get("seed"),
        "csv_path": _dataset_csv_path(dataset_entry),
        "model": model_name,
        "config_path": str(config_path).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/") if output_dir else "",
        "n_rows": summary.get("n_rows"),
        "n_series": summary.get("n_series"),
        "T_effective": summary.get("T_effective"),
        "mean_y": summary.get("mean_y"),
        "std_y": summary.get("std_y"),
    }
    return row


def run_synthetic_suite_benchmark(
    manifest_path: str = "data/simulated/manifest.json",
    models: Optional[Iterable[str]] = None,
    output_dir: str = "outputs/synthetic_suite",
    config_dir: str = "configs/generated/synthetic_suite",
    model_config_dir: Optional[str] = "configs/model",
    context_length: int = 96,
    prediction_length: int = 1,
    stride: int = 1,
    train_size: float = 0.6,
    val_size: float = 0.2,
    num_samples: Optional[int] = 100,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> SyntheticSuiteBenchmarkResult:
    """Run selected models over every dataset listed in a synthetic manifest."""

    manifest = _load_manifest(manifest_path)
    datasets = list(manifest["datasets"])
    model_names = _resolve_models(models)
    output_root = Path(output_dir)
    config_root = Path(config_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    result_csvs: Dict[str, Path] = {}
    result_jsons: Dict[str, Path] = {}

    for model_name in model_names:
        model_token = _safe_model_token(model_name)
        model_template = _load_model_template(model_token, model_config_dir)
        model_rows: List[Dict[str, Any]] = []
        for dataset_entry in datasets:
            dataset_id = _dataset_id(dataset_entry)
            config = build_synthetic_suite_config(
                dataset_entry=dataset_entry,
                model_name=model_token,
                model_template=model_template,
                output_dir=str(output_root),
                context_length=context_length,
                prediction_length=prediction_length,
                stride=stride,
                train_size=train_size,
                val_size=val_size,
                num_samples=num_samples,
                max_epochs=max_epochs,
                batch_size=batch_size,
                device=device,
            )
            config_path = config_root / model_token / f"{dataset_id}_{model_token}.yaml"
            save_yaml_config(config, str(config_path))
            row = _base_row(dataset_entry, model_token, config_path)

            if dry_run:
                row.update({"exit_code": 0, "status": "dry_run"})
            else:
                try:
                    result = run_experiment(str(config_path))
                    row["output_dir"] = str(result.output_dir).replace("\\", "/")
                    row.update(result.forecast_metrics)
                    row.update({"exit_code": 0, "status": "ok", "error": ""})
                except Exception as exc:  # pragma: no cover - covered by smoke/integration runs
                    row.update(
                        {
                            "exit_code": 1,
                            "status": "failed",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if not continue_on_error:
                        model_rows.append(row)
                        all_rows.append(row)
                        raise

            model_rows.append(row)
            all_rows.append(row)

        csv_path = output_root / f"results_{model_token}.csv"
        json_path = output_root / f"results_{model_token}.json"
        _write_rows_csv(model_rows, csv_path)
        _write_json(model_rows, json_path)
        result_csvs[model_token] = csv_path
        result_jsons[model_token] = json_path

    _write_json(
        {
            "manifest_path": manifest_path,
            "models": model_names,
            "output_dir": str(output_root),
            "config_dir": str(config_root),
            "num_datasets": len(datasets),
            "num_rows": len(all_rows),
            "result_csvs": {model: str(path) for model, path in result_csvs.items()},
        },
        output_root / "synthetic_suite_summary.json",
    )

    return SyntheticSuiteBenchmarkResult(
        output_dir=output_root,
        config_dir=config_root,
        result_csvs=result_csvs,
        result_jsons=result_jsons,
        rows=all_rows,
    )
