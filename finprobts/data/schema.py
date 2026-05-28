"""Canonical data containers used across the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _as_datetime_array(dates: Any) -> np.ndarray:
    parsed = pd.to_datetime(dates)
    return np.asarray(parsed, dtype="datetime64[ns]")


@dataclass
class FinancialDataset:
    """Canonical financial panel with shape ``[time, assets]``.

    Args:
        values: Target values, usually returns, with shape ``[T, N]``.
        dates: Timestamps with length ``T``.
        asset_ids: Asset identifiers with length ``N``.
        features: Optional feature arrays. Feature arrays should align on the
            time dimension and usually have shape ``[T, N]``.
        metadata: Free-form dataset metadata.
    """

    values: np.ndarray
    dates: Any
    asset_ids: List[str]
    features: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2:
            raise ValueError("FinancialDataset.values must have shape [time, assets].")

        self.dates = _as_datetime_array(self.dates)
        if len(self.dates) != self.values.shape[0]:
            raise ValueError("dates length must match the time dimension of values.")

        self.asset_ids = [str(asset_id) for asset_id in self.asset_ids]
        if len(self.asset_ids) != self.values.shape[1]:
            raise ValueError("asset_ids length must match the asset dimension of values.")

        if self.features is None:
            self.features = {}
        else:
            validated = {}
            for name, array in self.features.items():
                arr = np.asarray(array)
                if arr.shape[0] != self.values.shape[0]:
                    raise ValueError(f"Feature '{name}' must align with the time dimension.")
                validated[str(name)] = arr
            self.features = validated

    @property
    def num_timesteps(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_assets(self) -> int:
        return int(self.values.shape[1])

    def copy_with(
        self,
        values: Optional[np.ndarray] = None,
        dates: Optional[Any] = None,
        asset_ids: Optional[List[str]] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FinancialDataset":
        """Return a new dataset with selected fields replaced."""

        return FinancialDataset(
            values=self.values.copy() if values is None else values,
            dates=self.dates.copy() if dates is None else dates,
            asset_ids=list(self.asset_ids if asset_ids is None else asset_ids),
            features=dict(self.features if features is None else features),
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def slice_time(self, start: int, end: int) -> "FinancialDataset":
        """Return a chronological slice over ``[start, end)``."""

        features = {
            name: values[start:end].copy()
            for name, values in self.features.items()
        }
        metadata = dict(self.metadata)
        metadata["slice"] = {"start": int(start), "end": int(end)}
        return FinancialDataset(
            values=self.values[start:end].copy(),
            dates=self.dates[start:end].copy(),
            asset_ids=list(self.asset_ids),
            features=features,
            metadata=metadata,
        )


@dataclass
class TimeSeriesSplit:
    """Chronological train/validation/test split."""

    train: FinancialDataset
    val: FinancialDataset
    test: FinancialDataset


@dataclass
class RollingWindowDataset:
    """Canonical rolling-window forecasting task.

    ``x_context`` and ``y_target`` are the only arrays model adapters need to
    consume. Metadata preserves dates and asset IDs for fair evaluation.
    """

    x_context: np.ndarray
    y_target: np.ndarray
    context_dates: np.ndarray
    target_dates: np.ndarray
    asset_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_context = np.asarray(self.x_context, dtype=float)
        self.y_target = np.asarray(self.y_target, dtype=float)
        if self.x_context.ndim != 3:
            raise ValueError("x_context must have shape [num_windows, context_length, num_assets].")
        if self.y_target.ndim != 3:
            raise ValueError("y_target must have shape [num_windows, prediction_length, num_assets].")
        if self.x_context.shape[0] != self.y_target.shape[0]:
            raise ValueError("x_context and y_target must have the same number of windows.")
        if self.x_context.shape[2] != self.y_target.shape[2]:
            raise ValueError("x_context and y_target must have the same number of assets.")

        self.context_dates = np.asarray(self.context_dates, dtype="datetime64[ns]")
        self.target_dates = np.asarray(self.target_dates, dtype="datetime64[ns]")
        expected_context = self.x_context.shape[:2]
        expected_target = self.y_target.shape[:2]
        if self.context_dates.shape != expected_context:
            raise ValueError("context_dates must have shape [num_windows, context_length].")
        if self.target_dates.shape != expected_target:
            raise ValueError("target_dates must have shape [num_windows, prediction_length].")

        self.asset_ids = [str(asset_id) for asset_id in self.asset_ids]
        if len(self.asset_ids) != self.x_context.shape[2]:
            raise ValueError("asset_ids length must match the asset dimension.")

    def __len__(self) -> int:
        return int(self.x_context.shape[0])

    @property
    def num_assets(self) -> int:
        return int(self.x_context.shape[2])

    @property
    def context_length(self) -> int:
        return int(self.x_context.shape[1])

    @property
    def prediction_length(self) -> int:
        return int(self.y_target.shape[1])

    @property
    def start_dates(self) -> np.ndarray:
        """First forecast timestamp for each rolling window."""

        return self.target_dates[:, 0]
