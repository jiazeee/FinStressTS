from __future__ import annotations

import pandas as pd
import yaml

from finprobts.cli import main


def test_cli_config_smoke_run(tmp_path):
    data_path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=80),
            "a": [0.001 * i for i in range(80)],
            "b": [0.002 * i for i in range(80)],
        }
    ).to_csv(data_path, index=False)

    output_dir = tmp_path / "outputs"
    config = {
        "run": {"run_id": "smoke", "output_dir": str(output_dir), "seed": 123},
        "dataset": {
            "name": "custom_csv",
            "path": str(data_path),
            "format": "wide",
            "date_column": "date",
        },
        "preprocessing": {
            "value_kind": "returns",
            "missing_method": "ffill",
            "standardize": True,
        },
        "split": {"train_size": 0.5, "val_size": 0.25},
        "task": {"context_length": 10, "prediction_length": 3, "stride": 2},
        "model": {"name": "naive", "params": {}},
        "forecast": {"num_samples": 5},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == 0
    assert (output_dir / "smoke" / "forecast_samples.npz").exists()
    assert (output_dir / "smoke" / "forecast_metrics.json").exists()


def test_cli_generate_synthetic_smoke(tmp_path):
    out_dir = tmp_path / "synthetic"

    assert main([
        "generate-synthetic",
        "--case",
        "case1_garch",
        "--levels",
        "1",
        "--T",
        "30",
        "--n-firms",
        "3",
        "--out-dir",
        str(out_dir),
    ]) == 0
    assert (out_dir / "manifest.json").exists()


def test_cli_data_efficiency_smoke_run(tmp_path):
    data_path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=90),
            "a": [0.001 * i for i in range(90)],
            "b": [0.002 * i for i in range(90)],
        }
    ).to_csv(data_path, index=False)

    output_dir = tmp_path / "outputs"
    config = {
        "run": {"run_id": "efficiency_smoke", "output_dir": str(output_dir), "seed": 123},
        "dataset": {
            "name": "custom_csv",
            "path": str(data_path),
            "format": "wide",
            "date_column": "date",
        },
        "preprocessing": {
            "value_kind": "returns",
            "missing_method": "ffill",
            "standardize": True,
        },
        "split": {"train_size": 0.6, "val_size": 0.2},
        "task": {"context_length": 8, "prediction_length": 2, "stride": 2},
        "data_efficiency": {
            "train_fractions": [0.5, 1.0],
            "models": [{"name": "naive", "params": {}}],
        },
        "forecast": {"num_samples": 5},
    }
    config_path = tmp_path / "data_efficiency.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["data-efficiency", "--config", str(config_path)]) == 0
    results_csv = output_dir / "efficiency_smoke" / "data_efficiency_results.csv"
    results_json = output_dir / "efficiency_smoke" / "data_efficiency_results.json"
    assert results_csv.exists()
    assert results_json.exists()

    rows = pd.read_csv(results_csv)
    assert rows["train_fraction"].tolist() == [0.5, 1.0]
    assert rows["exit_code"].tolist() == [0, 0]
    assert rows["dataset_id"].tolist() == ["returns", "returns"]


def test_cli_data_efficiency_multi_dataset_plot_run(tmp_path):
    data_path_a = tmp_path / "returns_a.csv"
    data_path_b = tmp_path / "returns_b.csv"
    dates = pd.date_range("2024-01-01", periods=90)
    pd.DataFrame(
        {
            "date": dates,
            "a": [0.001 * i for i in range(90)],
            "b": [0.002 * i for i in range(90)],
        }
    ).to_csv(data_path_a, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "a": [0.0015 * i for i in range(90)],
            "b": [0.001 * i for i in range(90)],
        }
    ).to_csv(data_path_b, index=False)

    output_dir = tmp_path / "outputs"
    config = {
        "run": {"run_id": "efficiency_multi_plot", "output_dir": str(output_dir), "seed": 123},
        "datasets": [
            {
                "id": "dataset_a",
                "name": "custom_csv",
                "path": str(data_path_a),
                "format": "wide",
                "date_column": "date",
            },
            {
                "id": "dataset_b",
                "name": "custom_csv",
                "path": str(data_path_b),
                "format": "wide",
                "date_column": "date",
            },
        ],
        "preprocessing": {
            "value_kind": "returns",
            "missing_method": "ffill",
            "standardize": True,
        },
        "split": {"train_size": 0.6, "val_size": 0.2},
        "task": {"context_length": 8, "prediction_length": 2, "stride": 2},
        "data_efficiency": {
            "train_fractions": [0.5, 1.0],
            "models": [{"name": "naive", "params": {}}],
        },
        "forecast": {"num_samples": 5},
    }
    config_path = tmp_path / "data_efficiency_multi.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["data-efficiency", "--config", str(config_path), "--plot"]) == 0

    run_dir = output_dir / "efficiency_multi_plot"
    results_csv = run_dir / "data_efficiency_results.csv"
    plot_path = run_dir / "plots" / "data_efficiency_crps_naive.png"
    assert results_csv.exists()
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    rows = pd.read_csv(results_csv)
    assert sorted(rows["dataset_id"].unique().tolist()) == ["dataset_a", "dataset_b"]
    assert rows.shape[0] == 4
    assert rows["exit_code"].tolist() == [0, 0, 0, 0]


def test_cli_data_efficiency_config_plot_run(tmp_path):
    data_path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=90),
            "a": [0.001 * i for i in range(90)],
            "b": [0.002 * i for i in range(90)],
        }
    ).to_csv(data_path, index=False)

    output_dir = tmp_path / "outputs"
    config = {
        "run": {"run_id": "efficiency_config_plot", "output_dir": str(output_dir), "seed": 123},
        "dataset": {
            "id": "single_dataset",
            "name": "custom_csv",
            "path": str(data_path),
            "format": "wide",
            "date_column": "date",
        },
        "preprocessing": {
            "value_kind": "returns",
            "missing_method": "ffill",
            "standardize": True,
        },
        "split": {"train_size": 0.6, "val_size": 0.2},
        "task": {"context_length": 8, "prediction_length": 2, "stride": 2},
        "data_efficiency": {
            "train_fractions": [1.0],
            "plot": True,
            "models": [{"name": "naive", "params": {}}],
        },
        "forecast": {"num_samples": 5},
    }
    config_path = tmp_path / "data_efficiency_plot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["data-efficiency", "--config", str(config_path)]) == 0
    plot_path = output_dir / "efficiency_config_plot" / "plots" / "data_efficiency_crps_naive.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
