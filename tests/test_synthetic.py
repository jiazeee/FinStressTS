from __future__ import annotations

import json

import finprobts
from finprobts.data import get_default_dataset_registry
from finprobts.simulators import (
    BaseSimulator,
    GARCHSimulator,
    HARSimulator,
    HeavyTailSimulator,
    MarketHawkesPanelSimulator,
    MarketRegimePanelSimulator,
    MarketZIPPanelSimulator,
)
from finprobts.synthetic import generate_synthetic_case, generate_synthetic_suite


def test_simulators_import_from_package():
    assert BaseSimulator is not None
    assert GARCHSimulator is not None
    assert HARSimulator is not None
    assert HeavyTailSimulator is not None
    assert MarketRegimePanelSimulator is not None
    assert MarketHawkesPanelSimulator is not None
    assert MarketZIPPanelSimulator is not None


def test_root_package_imports_after_legacy_move():
    assert finprobts.FinancialDataset is not None


def test_synthetic_registry_uses_packaged_simulators():
    dataset = get_default_dataset_registry().load("synthetic_garch", T=30, n_firms=3, seed=1)

    assert dataset.values.shape == (30, 3)
    assert dataset.metadata["dataset_name"] == "synthetic_garch"


def test_generate_synthetic_case_writes_artifacts(tmp_path):
    out = generate_synthetic_case(
        case="case1_garch",
        level=1,
        out_dir=str(tmp_path),
        base_seed=10,
        T=30,
        n_firms=3,
        formats=("csv",),
    )

    assert (tmp_path / f"{out['tag']}.csv").exists()
    assert (tmp_path / f"{out['tag']}.meta.json").exists()
    assert out["summary"]["n_series"] == 3
    with open(out["meta_path"], "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    assert meta["case"] == "case1_garch"


def test_generate_synthetic_suite_writes_manifest(tmp_path):
    manifest = generate_synthetic_suite(
        case="case1_garch",
        levels=(1,),
        out_dir=str(tmp_path),
        base_seed=10,
        T=30,
        n_firms=3,
        formats=("csv",),
    )

    assert len(manifest["datasets"]) == 1
    assert (tmp_path / "manifest.json").exists()
