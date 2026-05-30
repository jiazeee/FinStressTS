"""Dataset registry for named benchmark data sources."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from finprobts.data.io import load_financial_data
from finprobts.data.schema import FinancialDataset


DatasetLoader = Callable[..., FinancialDataset]


class DatasetRegistry:
    """Simple name-to-loader registry for datasets."""

    def __init__(self) -> None:
        self._loaders: Dict[str, DatasetLoader] = {}

    def register(self, name: str, loader: DatasetLoader) -> None:
        if not name:
            raise ValueError("Dataset name must be non-empty.")
        self._loaders[name] = loader

    def get(self, name: str) -> DatasetLoader:
        try:
            return self._loaders[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._loaders)) or "<none>"
            raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc

    def load(self, name: str, **kwargs) -> FinancialDataset:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._loaders)


def _custom_csv_loader(**kwargs) -> FinancialDataset:
    kwargs = dict(kwargs)
    kwargs.setdefault("format", kwargs.pop("data_format", "wide"))
    return load_financial_data(**kwargs)


def _custom_parquet_loader(**kwargs) -> FinancialDataset:
    kwargs = dict(kwargs)
    kwargs.setdefault("format", kwargs.pop("data_format", "wide"))
    return load_financial_data(**kwargs)


def _djia_loader(**kwargs) -> FinancialDataset:
    path = kwargs.pop(
        "path",
        "data/historical_stock_data/dj30_returns_20160101_to_20260101_wide.csv",
    )
    target_columns = kwargs.pop("target_columns", None)
    dataset = load_financial_data(
        path=path,
        format="wide",
        date_column=kwargs.pop("date_column", "date"),
        target_columns=target_columns,
        metadata={"dataset_name": "djia"},
    )
    if "DOW" in dataset.asset_ids:
        keep = [idx for idx, asset in enumerate(dataset.asset_ids) if asset != "DOW"]
        dataset = dataset.copy_with(
            values=dataset.values[:, keep],
            asset_ids=[dataset.asset_ids[idx] for idx in keep],
        )
    return dataset


def _synthetic_loader(simulator_name: str) -> DatasetLoader:
    def load_synthetic(**kwargs) -> FinancialDataset:
        simulator_kwargs = dict(kwargs)
        simulator_kwargs.pop("format", None)
        simulator_kwargs.pop("data_format", None)

        if simulator_name == "garch":
            from finprobts.simulators.garch import GARCHSimulator

            sim = GARCHSimulator(**simulator_kwargs)
        elif simulator_name == "har":
            from finprobts.simulators.har import HARSimulator

            sim = HARSimulator(**simulator_kwargs)
        elif simulator_name == "heavy_tail":
            from finprobts.simulators.heavy_tail import HeavyTailSimulator

            sim = HeavyTailSimulator(**simulator_kwargs)
        elif simulator_name == "regime":
            from finprobts.simulators.regime_switching import MarketRegimePanelSimulator

            sim = MarketRegimePanelSimulator(**simulator_kwargs)
        elif simulator_name == "hawkes":
            from finprobts.simulators.hawkes import MarketHawkesPanelSimulator

            sim = MarketHawkesPanelSimulator(**simulator_kwargs)
        elif simulator_name == "zip":
            from finprobts.simulators.zero_inflated import MarketZIPPanelSimulator

            sim = MarketZIPPanelSimulator(**simulator_kwargs)
        else:
            raise ValueError(f"Unknown synthetic simulator '{simulator_name}'.")

        result = sim.simulate()
        values = result.get("returns", result.get("y"))
        if values is None:
            raise ValueError(f"Simulator '{simulator_name}' did not return returns or y.")
        values = np.asarray(values, dtype=float)
        return FinancialDataset(
            values=values,
            dates=np.arange(values.shape[0], dtype="int64"),
            asset_ids=[f"asset_{idx}" for idx in range(values.shape[1])],
            metadata={
                "dataset_name": f"synthetic_{simulator_name}",
                "simulator_params": result.get("params", {}),
                "time_index_kind": "relative",
                "freq": "D",
            },
        )

    return load_synthetic


def get_default_dataset_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register("custom_csv", _custom_csv_loader)
    registry.register("custom_parquet", _custom_parquet_loader)
    registry.register("djia", _djia_loader)
    for name in ["garch", "har", "heavy_tail", "regime", "hawkes", "zip"]:
        registry.register(f"synthetic_{name}", _synthetic_loader(name))
    return registry
