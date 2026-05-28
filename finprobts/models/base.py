"""Canonical probabilistic forecasting model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from finprobts.data.schema import RollingWindowDataset


@dataclass
class ForecastResult:
    """Standard forecast output consumed by evaluation and backtesting.

    Args:
        samples: Forecast samples with shape
            ``[num_windows, num_samples, prediction_length, num_assets]``.
        y_true: Realized values with shape
            ``[num_windows, prediction_length, num_assets]``.
        start_dates: First target date for each window.
        item_ids: Asset identifiers.
        metadata: Free-form run/model metadata.
    """

    samples: np.ndarray
    y_true: np.ndarray
    start_dates: Any
    item_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        self.y_true = np.asarray(self.y_true, dtype=float)
        if self.samples.ndim != 4:
            raise ValueError(
                "samples must have shape [num_windows, num_samples, prediction_length, num_assets]."
            )
        if self.y_true.ndim != 3:
            raise ValueError("y_true must have shape [num_windows, prediction_length, num_assets].")
        if self.samples.shape[0] != self.y_true.shape[0]:
            raise ValueError("samples and y_true must have the same number of windows.")
        if self.samples.shape[2:] != self.y_true.shape[1:]:
            raise ValueError("samples prediction_length/assets must match y_true.")

        self.start_dates = np.asarray(self.start_dates, dtype="datetime64[ns]")
        if len(self.start_dates) != self.samples.shape[0]:
            raise ValueError("start_dates length must match num_windows.")

        self.item_ids = [str(item_id) for item_id in self.item_ids]
        if len(self.item_ids) != self.samples.shape[3]:
            raise ValueError("item_ids length must match num_assets.")

    @property
    def num_windows(self) -> int:
        return int(self.samples.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[1])

    @property
    def prediction_length(self) -> int:
        return int(self.samples.shape[2])

    @property
    def num_assets(self) -> int:
        return int(self.samples.shape[3])

    def sample_mean(self) -> np.ndarray:
        """Return point forecasts from the sample mean."""

        return self.samples.mean(axis=1)

    def with_arrays(self, samples: np.ndarray, y_true: np.ndarray) -> "ForecastResult":
        """Return a copy with replaced forecast arrays."""

        return ForecastResult(
            samples=samples,
            y_true=y_true,
            start_dates=self.start_dates.copy(),
            item_ids=list(self.item_ids),
            metadata=dict(self.metadata),
        )

    def save_npz(self, path: str) -> None:
        """Persist forecast arrays and metadata-friendly identifiers."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            samples=self.samples,
            y_true=self.y_true,
            start_dates=self.start_dates.astype("datetime64[ns]").astype(str),
            item_ids=np.asarray(self.item_ids, dtype=str),
        )

    @classmethod
    def load_npz(cls, path: str, metadata: Optional[Dict[str, Any]] = None) -> "ForecastResult":
        loaded = np.load(path, allow_pickle=False)
        return cls(
            samples=loaded["samples"],
            y_true=loaded["y_true"],
            start_dates=loaded["start_dates"],
            item_ids=[str(item) for item in loaded["item_ids"]],
            metadata=dict(metadata or {}),
        )


class BaseProbForecastModel(ABC):
    """Abstract base class for benchmark-compatible probabilistic models."""

    @abstractmethod
    def fit(self, train_data: RollingWindowDataset, val_data: Optional[RollingWindowDataset] = None) -> None:
        """Fit the model to canonical rolling-window data."""

    @abstractmethod
    def predict(self, test_data: RollingWindowDataset, num_samples: int) -> ForecastResult:
        """Generate probabilistic forecasts in the canonical result format."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model state to a file or directory."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseProbForecastModel":
        """Load model state from a file or directory."""
