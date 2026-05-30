"""Preprocessing utilities for financial forecasting tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

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


def _dataframe_from_feature(array: np.ndarray, index: np.ndarray) -> tuple[pd.DataFrame, tuple[int, ...]]:
    arr = np.asarray(array)
    trailing_shape = arr.shape[1:]
    return pd.DataFrame(arr.reshape(arr.shape[0], -1), index=index), trailing_shape


def _ffill_with_initial(df: pd.DataFrame, initial_values: Optional[np.ndarray] = None) -> pd.DataFrame:
    if initial_values is None:
        return df.ffill()

    initial = np.asarray(initial_values, dtype=float).reshape(1, -1)
    if initial.shape[1] != df.shape[1]:
        raise ValueError("initial_values must have the same number of columns as the dataset.")
    seed = pd.DataFrame(initial, columns=df.columns)
    combined = pd.concat([seed, df.reset_index(drop=True)], axis=0, ignore_index=True)
    filled = combined.ffill().iloc[1:].copy()
    filled.index = df.index
    return filled


def handle_missing_values(
    dataset: FinancialDataset,
    method: str = "ffill",
    initial_values: Optional[np.ndarray] = None,
    initial_features: Optional[Dict[str, np.ndarray]] = None,
    allow_leading_backfill: bool = False,
) -> FinancialDataset:
    """Handle missing values in the dataset target array.

    Supported methods are ``ffill``, ``bfill``, ``drop``, ``zero``, and
    ``none``. ``ffill`` is causal by default: it only uses past observations
    and optional ``initial_values`` from a previous chronological split. Set
    ``allow_leading_backfill=True`` only when future-looking imputation is an
    explicit experimental choice.
    """

    method = method.lower()
    df = pd.DataFrame(dataset.values, index=dataset.dates, columns=dataset.asset_ids)

    if method == "none":
        values = df.to_numpy(dtype=float)
        dates = dataset.dates.copy()
    elif method == "ffill":
        filled = _ffill_with_initial(df, initial_values=initial_values)
        if allow_leading_backfill:
            filled = filled.bfill()
        values = filled.to_numpy(dtype=float)
        dates = dataset.dates.copy()
    elif method == "bfill":
        filled = df.bfill()
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

    if method != "none" and np.isnan(values).any():
        raise ValueError(
            "Missing values remain after preprocessing. For causal ffill this usually means "
            "a series starts with missing values; use drop/zero, provide prior split values, "
            "or explicitly set allow_leading_backfill=True."
        )

    features = {}
    for name, array in dataset.features.items():
        feature_df, trailing_shape = _dataframe_from_feature(array, dataset.dates)
        initial_feature = None if initial_features is None else initial_features.get(name)
        if initial_feature is not None:
            initial_feature = np.asarray(initial_feature).reshape(1, -1)
        if method == "drop":
            feature_df = feature_df.reindex(dates)
        elif method == "ffill":
            feature_df = _ffill_with_initial(feature_df, initial_values=initial_feature)
            if allow_leading_backfill:
                feature_df = feature_df.bfill()
        elif method == "bfill":
            feature_df = feature_df.bfill()
        elif method == "zero":
            feature_df = feature_df.fillna(0.0)
        feature_values = feature_df.to_numpy()
        if method != "none" and np.isnan(feature_values).any():
            raise ValueError(f"Missing values remain in feature '{name}' after preprocessing.")
        features[name] = feature_values.reshape((feature_values.shape[0], *trailing_shape))

    metadata = dict(dataset.metadata)
    metadata["missing_value_method"] = method
    return FinancialDataset(
        values=values,
        dates=dates,
        asset_ids=list(dataset.asset_ids),
        features=features,
        metadata=metadata,
    )


def handle_missing_values_split_safe(
    split: TimeSeriesSplit,
    method: str = "ffill",
    allow_leading_backfill: bool = False,
) -> TimeSeriesSplit:
    """Apply missing-value handling without leaking future splits backward."""

    method = method.lower()
    if method != "ffill":
        return TimeSeriesSplit(
            train=handle_missing_values(split.train, method=method, allow_leading_backfill=allow_leading_backfill),
            val=handle_missing_values(split.val, method=method, allow_leading_backfill=allow_leading_backfill),
            test=handle_missing_values(split.test, method=method, allow_leading_backfill=allow_leading_backfill),
        )

    train = handle_missing_values(
        split.train,
        method=method,
        allow_leading_backfill=allow_leading_backfill,
    )
    val = handle_missing_values(
        split.val,
        method=method,
        initial_values=train.values[-1],
        initial_features={name: values[-1] for name, values in train.features.items()},
        allow_leading_backfill=allow_leading_backfill,
    )
    test = handle_missing_values(
        split.test,
        method=method,
        initial_values=val.values[-1],
        initial_features={name: values[-1] for name, values in val.features.items()},
        allow_leading_backfill=allow_leading_backfill,
    )
    return TimeSeriesSplit(train=train, val=val, test=test)


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
        values = np.asarray(dataset.values, dtype=float)
        finite = np.isfinite(values)
        if np.any(finite.sum(axis=0) == 0):
            missing_assets = [
                asset_id
                for asset_id, count in zip(dataset.asset_ids, finite.sum(axis=0))
                if int(count) == 0
            ]
            raise ValueError(
                "Cannot fit DatasetNormalizer because some assets have no finite "
                f"training values: {missing_assets}"
            )

        cleaned = np.where(finite, values, np.nan)
        mean = np.nanmean(cleaned, axis=0)
        std = np.nanstd(cleaned, axis=0)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("DatasetNormalizer fitted non-finite statistics.")
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
