from __future__ import annotations

import json
from pathlib import Path

import yaml

from finprobts.experiment.synthetic_suite import (
    build_synthetic_suite_config,
    run_synthetic_suite_benchmark,
)


def _manifest(tmp_path: Path) -> Path:
    csv_path = tmp_path / "case1_garch_level01_N3_T30_seed124.csv"
    csv_path.write_text("time,series_id,y\n0,asset_0,0.0\n", encoding="utf-8")
    manifest = {
        "datasets": [
            {
                "case": "case1_garch",
                "level": 1,
                "tag": "case1_garch_level01_N3_T30_seed124",
                "seed": 124,
                "csv_path": str(csv_path),
                "summary": {"n_rows": 90, "n_series": 3, "T_effective": 30},
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_build_synthetic_suite_config_uses_long_dataset_contract(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    dataset_entry = json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"][0]

    config = build_synthetic_suite_config(
        dataset_entry=dataset_entry,
        model_name="deepvar",
        model_template={"model": {"name": "deepvar", "params": {"hidden_size": 8}}},
        output_dir=str(tmp_path / "outputs"),
        num_samples=7,
        max_epochs=1,
        device="cpu",
    )

    assert config["dataset"]["format"] == "long"
    assert config["dataset"]["date_column"] == "time"
    assert config["dataset"]["asset_id_column"] == "series_id"
    assert config["dataset"]["target_column"] == "y"
    assert config["task"]["context_length"] == 96
    assert config["task"]["prediction_length"] == 1
    assert config["model"]["params"]["max_epochs"] == 1
    assert config["model"]["params"]["device"] == "cpu"
    assert config["forecast"]["num_samples"] == 7


def test_run_synthetic_suite_dry_run_writes_configs_and_model_csv(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    output_dir = tmp_path / "outputs"
    config_dir = tmp_path / "configs"

    result = run_synthetic_suite_benchmark(
        manifest_path=str(manifest_path),
        models=["naive"],
        output_dir=str(output_dir),
        config_dir=str(config_dir),
        model_config_dir=None,
        dry_run=True,
    )

    assert len(result.rows) == 1
    assert result.rows[0]["status"] == "dry_run"
    assert "naive" in result.result_csvs
    assert result.result_csvs["naive"].exists()
    generated_configs = list((config_dir / "naive").glob("*.yaml"))
    assert len(generated_configs) == 1

    config = yaml.safe_load(generated_configs[0].read_text(encoding="utf-8"))
    assert config["model"]["name"] == "naive"
    assert config["dataset"]["format"] == "long"
