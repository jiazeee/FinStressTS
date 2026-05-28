"""Preprocessing utilities for financial forecasting tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from finprobts.data.schema import FinancialDataset, TimeSeriesSplit


ArrayLike = Union[np.ndarray, pd.DataFrame, FinancialDataset]


def _log_return_values(values: np.ndarray) -> np.ndarray:
    if np.any(values <= 0):
        raise ValueError("price_to_log_return requires strictly positive prices.")
    return np.diff(np.log(values), axis=0)


def price_to_log_return(data: ArrayLike) -> ArrayLike:
    """Convert prices to log returns, dropping the first timestamp.

    The return type matches the input type for ``FinancialDataset``, NumPy
    arrays, and pandas DataFrames.
    """

    if isinstance(data, FinancialDataset):
        values = _log_return_values(data.values)
        features = {
            name: array[1:].copy()
            for name, array in data.features.items()
        }
        metadata = dict(data.metadata)
        metadata["value_kind"] = "log_return"
        return FinancialDataset(
            values=values,
            dates=data.dates[1:].copy(),
            asset_ids=list(data.asset_ids),
            features=features,
            metadata=metadata,
        )

    if isinstance(data, pd.DataFrame):
        returns = np.diff(np.log(data.to_numpy(dtype=float)), axis=0)
        return pd.DataFrame(returns, index=data.index[1:], columns=data.columns)

    return _log_return_values(np.asarray(data, dtype=float))


def handle_missing_values(dataset: FinancialDataset, method: str = "ffill") -> FinancialDataset:
    """Handle missing values in the dataset target array.

    Supported methods are ``ffill``, ``bfill``, ``drop``, ``zero``, and
    ``none``. ``ffill`` also backfills leading gaps so the output contains no
    missing values when each column has at least one observation.
    """

    method = method.lower()
    df = pd.DataFrame(dataset.values, index=dataset.dates, columns=dataset.asset_ids)

    if method == "none":
        values = df.to_numpy(dtype=float)
        dates = dataset.dates.copy()
    elif method == "ffill":
        filled = df.ffill().bfill()
        values = filled.to_numpy(dtype=float)
        dates = dataset.dates.copy()
    elif method == "bfill":
        filled = df.bfill().ffill()
        values = filled.to_numpy(dtype=float)
        dates = dataset.dates.copy()
    elif method == "zero":
        values = df.fillna(0.0).to_numpy(dtype=float)
        dates = dataset.dates.copy()
    elif method == "drop":
        dropped = df.dropna(axis=0, how="any")
        values = dropped.to_numpy(dtype=float)
        dates = np.asarray(dropped.index, dtype="datetime64[ns]")
    else:
        raise ValueError("method must be one of: ffill, bfill, drop, zero, none.")

    if np.isnan(values).any():
        raise ValueError("Missing values remain after preprocessing.")

    features = {}
    for name, array in dataset.features.items():
        feature_df = pd.DataFrame(array, index=dataset.dates)
        if method == "drop":
            feature_df = feature_df.reindex(dates)
        elif method == "ffill":
            feature_df = feature_df.ffill().bfill()
        elif method == "bfill":
            feature_df = feature_df.bfill().ffill()
        elif method == "zero":
            feature_df = feature_df.fillna(0.0)
        features[name] = feature_df.to_numpy()

    metadata = dict(dataset.metadata)
    metadata["missing_value_method"] = method
    return FinancialDataset(
        values=values,
        dates=dates,
        asset_ids=list(dataset.asset_ids),
        features=features,
        metadata=metadata,
    )


def rolling_normalize(
    values: np.ndarray,
    window: int,
    min_periods: int = 1,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply trailing rolling z-score normalization to an array.

    Returns ``normalized, rolling_mean, rolling_std``. The rolling statistics
    include the current timestamp, which is useful for exploratory
    preprocessing. Train-fitted standardization is used by the experiment
    runner for benchmark fairness.
    """

    if window <= 0:
        raise ValueError("window must be positive.")
    if min_periods <= 0:
        raise ValueError("min_periods must be positive.")

    df = pd.DataFrame(np.asarray(values, dtype=float))
    rolling = df.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean().to_numpy()
    std = rolling.std(ddof=0).to_numpy()
    std = np.where(np.isnan(std) | (std < eps), 1.0, std)
    normalized = (df.to_numpy() - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized, mean, std


@dataclass
class DatasetNormalizer:
    """Train-fitted per-asset standardizer."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, dataset: FinancialDataset, eps: float = 1e-8) -> "DatasetNormalizer":
        mean = dataset.values.mean(axis=0)
        std = dataset.values.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_transform_values(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean

    def transform_dataset(self, dataset: FinancialDataset) -> FinancialDataset:
        metadata = dict(dataset.metadata)
        metadata["standardized"] = True
        return dataset.copy_with(
            values=self.transform_values(dataset.values),
            metadata=metadata,
        )

    def inverse_transform_samples(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(samples, dtype=float) * self.std.reshape(1, 1, 1, -1) + self.mean.reshape(1, 1, 1, -1)

    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return np.asarray(targets, dtype=float) * self.std.reshape(1, 1, -1) + self.mean.reshape(1, 1, -1)


def _split_sizes(
    n: int,
    train_size: float,
    val_size: float,
    test_size: Optional[float],
) -> Tuple[int, int]:
    if not (0.0 < train_size < 1.0):
        raise ValueError("train_size must be a fraction in (0, 1).")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be a fraction in [0, 1).")
    if test_size is not None and not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a fraction in (0, 1).")

    total = train_size + val_size + (0.0 if test_size is None else test_size)
    if test_size is None:
        if train_size + val_size >= 1.0:
            raise ValueError("train_size + val_size must be less than 1.")
    elif not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.")

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Split fractions produce an empty train, validation, or test set.")
    return train_end, val_end


def time_train_val_test_split(
    dataset: FinancialDataset,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: Optional[float] = None,
) -> TimeSeriesSplit:
    """Split a dataset chronologically into train, validation, and test sets."""

    train_end, val_end = _split_sizes(dataset.num_timesteps, train_size, val_size, test_size)
    return TimeSeriesSplit(
        train=dataset.slice_time(0, train_end),
        val=dataset.slice_time(train_end, val_end),
        test=dataset.slice_time(val_end, dataset.num_timesteps),
    )
