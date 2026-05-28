"""Shared utilities for native PyTorch forecasting models.

PyTorch is an optional dependency for FinProbTS-Bench. This module avoids
importing it at module import time so the core package still works in minimal
installations.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from finprobts.data.schema import RollingWindowDataset


BATCH_FIELDS = (
    "past_target",
    "future_target",
    "past_observed_values",
    "future_observed_values",
    "target_dimension_indicator",
    "past_time_feat",
    "future_time_feat",
)


def require_torch() -> Any:
    """Import torch or raise a helpful optional-dependency error."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "Native deep forecasting models require PyTorch. Install it with "
            "`pip install -e .[torch]` or install `torch` in your environment."
        ) from exc
    return torch


def set_torch_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and PyTorch RNGs when a seed is provided."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - depends on host hardware
        torch.cuda.manual_seed_all(int(seed))


def resolve_torch_device(device: Optional[str] = "auto") -> Any:
    """Resolve a PyTorch device string."""

    torch = require_torch()
    requested = "auto" if device is None else str(device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TorchStandardScaler:
    """Train-fitted standard scaler for tensors with assets on the last axis."""

    mean: np.ndarray
    std: np.ndarray
    min_std: float = 1e-6

    @classmethod
    def fit(cls, values: np.ndarray, min_std: float = 1e-6, var_specific: bool = True) -> "TorchStandardScaler":
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError("Cannot fit scaler on an empty array.")
        if var_specific:
            mean = np.nanmean(arr, axis=tuple(range(arr.ndim - 1)), keepdims=False)
            std = np.nanstd(arr, axis=tuple(range(arr.ndim - 1)), keepdims=False)
        else:
            mean = np.asarray(np.nanmean(arr), dtype=float)
            std = np.asarray(np.nanstd(arr), dtype=float)
        std = np.where(np.asarray(std) < min_std, min_std, std)
        return cls(mean=np.asarray(mean, dtype=float), std=np.asarray(std, dtype=float), min_std=float(min_std))

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_transform_array(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": np.asarray(self.mean, dtype=float).tolist(),
            "std": np.asarray(self.std, dtype=float).tolist(),
            "min_std": self.min_std,
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, Any]) -> "TorchStandardScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=float),
            std=np.asarray(payload["std"], dtype=float),
            min_std=float(payload.get("min_std", 1e-6)),
        )


def calendar_time_features(dates: np.ndarray) -> np.ndarray:
    """Return simple calendar features for date arrays.

    The output shape is ``dates.shape + (4,)`` and contains normalized month,
    day-of-month, day-of-week, and day-of-year features. Models can ignore these
    features, but the canonical torch batch includes them for future adapters.
    """

    arr = np.asarray(dates, dtype="datetime64[ns]")
    flat = arr.reshape(-1)
    index = pd.DatetimeIndex(flat)
    features = np.stack(
        [
            (index.month.to_numpy(dtype=float) - 1.0) / 11.0,
            (index.day.to_numpy(dtype=float) - 1.0) / 30.0,
            index.dayofweek.to_numpy(dtype=float) / 6.0,
            (index.dayofyear.to_numpy(dtype=float) - 1.0) / 365.0,
        ],
        axis=-1,
    )
    return features.reshape(arr.shape + (features.shape[-1],)).astype(np.float32)


def make_window_arrays(
    windows: RollingWindowDataset,
    scaler: Optional[TorchStandardScaler] = None,
    include_time_features: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert canonical rolling windows into model-ready NumPy arrays."""

    past = np.asarray(windows.x_context, dtype=np.float32)
    future = np.asarray(windows.y_target, dtype=np.float32)
    past_observed = np.isfinite(past).astype(np.float32)
    future_observed = np.isfinite(future).astype(np.float32)
    past = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
    future = np.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is not None:
        past = scaler.transform_array(past).astype(np.float32)
        future = scaler.transform_array(future).astype(np.float32)

    num_windows = len(windows)
    target_dim = np.tile(np.arange(windows.num_assets, dtype=np.int64), (num_windows, 1))
    if include_time_features:
        past_time_feat = calendar_time_features(windows.context_dates)
        future_time_feat = calendar_time_features(windows.target_dates)
    else:
        past_time_feat = np.zeros((*past.shape[:2], 0), dtype=np.float32)
        future_time_feat = np.zeros((*future.shape[:2], 0), dtype=np.float32)

    return {
        "past_target": past,
        "future_target": future,
        "past_observed_values": past_observed,
        "future_observed_values": future_observed,
        "target_dimension_indicator": target_dim,
        "past_time_feat": past_time_feat,
        "future_time_feat": future_time_feat,
    }


def make_torch_data_loader(
    windows: RollingWindowDataset,
    batch_size: int,
    shuffle: bool,
    scaler: Optional[TorchStandardScaler] = None,
    include_time_features: bool = True,
) -> Any:
    """Create a ``torch.utils.data.DataLoader`` from rolling windows."""

    torch = require_torch()
    arrays = make_window_arrays(windows, scaler=scaler, include_time_features=include_time_features)
    tensors = []
    for name in BATCH_FIELDS:
        dtype = torch.long if name == "target_dimension_indicator" else torch.float32
        tensors.append(torch.as_tensor(arrays[name], dtype=dtype))
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle))


def iter_torch_batches(loader: Any, device: Any) -> Iterator[Dict[str, Any]]:
    """Yield dictionary batches from a tensor dataloader."""

    for batch in loader:
        yield {
            name: tensor.to(device)
            for name, tensor in zip(BATCH_FIELDS, batch)
        }


def dump_jsonable(payload: Dict[str, Any]) -> str:
    """Serialize a small JSON-compatible payload for checkpoint metadata."""

    return json.dumps(payload, sort_keys=True)
