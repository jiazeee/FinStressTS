"""CSV and Parquet loaders for financial time-series panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from finprobts.data.schema import FinancialDataset


def _read_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(table_path)
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet files requires a pandas Parquet engine. "
                "Install finprobts-bench[parquet] or install pyarrow."
            ) from exc
    raise ValueError(f"Unsupported file extension '{suffix}'. Use CSV or Parquet.")


def _is_numeric_time(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    try:
        pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError):
        return False
    return True


def _validate_regular_relative(values: np.ndarray, date_column: str) -> None:
    unique = np.sort(np.unique(values.astype(float)))
    if unique.size < 3:
        return
    diffs = np.diff(unique)
    if np.any(diffs <= 0) or not np.allclose(diffs, 1.0):
        raise ValueError(
            f"Relative time column '{date_column}' must be regular. "
            "Expected consecutive integer-like steps. "
            "Set validate_regular=False only after deliberately handling irregular sampling."
        )


def _validate_regular_datetime(index: pd.DatetimeIndex, freq: Optional[str], date_column: str) -> Tuple[Optional[str], bool]:
    unique = pd.DatetimeIndex(index.unique()).sort_values()
    if len(unique) < 3:
        return freq, True

    if freq:
        expected = pd.date_range(start=unique[0], periods=len(unique), freq=str(freq))
        if not np.array_equal(unique.to_numpy(dtype="datetime64[ns]"), expected.to_numpy(dtype="datetime64[ns]")):
            raise ValueError(
                f"Datetime column '{date_column}' is not regular at configured freq='{freq}'. "
                "Regularize/resample the data or set validate_regular=False explicitly."
            )
        return str(freq), True

    inferred = pd.infer_freq(unique)
    if inferred is None:
        raise ValueError(
            f"Could not infer a regular frequency for datetime column '{date_column}'. "
            "Provide dataset.freq, regularize the data, or set validate_regular=False explicitly."
        )
    return str(inferred), True


def _relative_values_to_timestamps(unique_values: np.ndarray, freq: str, time_origin: str) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(freq)
    steps = unique_values.astype(float) - float(unique_values[0])
    if not np.all(np.isfinite(steps)):
        raise ValueError("Relative time values must be finite.")

    try:
        nanos = offset.nanos
    except ValueError as exc:
        rounded = np.round(steps).astype(int)
        if not np.allclose(steps, rounded):
            raise ValueError(
                f"Irregular relative time with non-fixed freq='{freq}' requires integer steps."
            ) from exc
        return pd.DatetimeIndex([pd.Timestamp(time_origin) + int(step) * offset for step in rounded])

    deltas = pd.to_timedelta(steps * float(nanos), unit="ns")
    return pd.DatetimeIndex(pd.Timestamp(time_origin) + deltas)


def _coerce_time_column(
    series: pd.Series,
    *,
    date_column: str,
    freq: Optional[str],
    time_index: str,
    validate_regular: bool,
    time_origin: str,
) -> Tuple[pd.Series, Dict[str, Any]]:
    if series.isna().any():
        raise ValueError(f"date_column '{date_column}' contains missing timestamps.")

    mode = str(time_index or "auto").lower()
    if mode not in {"auto", "datetime", "relative"}:
        raise ValueError("time_index must be one of: auto, datetime, relative.")
    if mode == "auto":
        mode = "relative" if _is_numeric_time(series) else "datetime"

    if mode == "relative":
        numeric = pd.to_numeric(series, errors="raise").astype(float)
        if validate_regular:
            _validate_regular_relative(numeric.to_numpy(), date_column)
        unique_values = np.sort(numeric.unique())
        relative_freq = str(freq or "D")
        timestamps = (
            pd.date_range(start=pd.Timestamp(time_origin), periods=len(unique_values), freq=relative_freq)
            if validate_regular
            else _relative_values_to_timestamps(unique_values, relative_freq, time_origin)
        )
        value_to_timestamp = dict(zip(unique_values, timestamps))
        return (
            numeric.map(value_to_timestamp),
            {
                "time_index_kind": "relative",
                "freq": relative_freq,
                "time_origin": str(time_origin),
                "time_index_is_regular": bool(validate_regular),
                "time_column": date_column,
            },
        )

    parsed = pd.to_datetime(series, errors="raise")
    resolved_freq = freq
    is_regular = False
    if validate_regular:
        resolved_freq, is_regular = _validate_regular_datetime(pd.DatetimeIndex(parsed), freq, date_column)
    else:
        inferred = pd.infer_freq(pd.DatetimeIndex(parsed).unique().sort_values()) if len(parsed.unique()) >= 3 else None
        resolved_freq = str(freq or inferred) if (freq or inferred) else None
    meta: Dict[str, Any] = {
        "time_index_kind": "datetime",
        "time_index_is_regular": is_regular,
        "time_column": date_column,
    }
    if resolved_freq:
        meta["freq"] = str(resolved_freq)
    return parsed, meta


def _resolve_target_columns(
    columns: Iterable[str],
    date_column: str,
    target_columns: Optional[List[str]],
) -> List[str]:
    if target_columns:
        return [str(col) for col in target_columns]
    return [str(col) for col in columns if str(col) != date_column]


def _load_wide(
    df: pd.DataFrame,
    date_column: str,
    target_columns: Optional[List[str]],
    metadata: Optional[dict],
    freq: Optional[str],
    time_index: str,
    validate_regular: bool,
    validate_unique_index: bool,
    time_origin: str,
) -> FinancialDataset:
    if date_column not in df.columns:
        raise ValueError(f"date_column '{date_column}' not found in data.")

    targets = _resolve_target_columns(df.columns, date_column, target_columns)
    missing = [col for col in targets if col not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found: {missing}")

    df = df[[date_column] + targets].copy()
    df[date_column], time_metadata = _coerce_time_column(
        df[date_column],
        date_column=date_column,
        freq=freq,
        time_index=time_index,
        validate_regular=validate_regular,
        time_origin=time_origin,
    )
    if validate_unique_index and df[date_column].duplicated().any():
        duplicates = df.loc[df[date_column].duplicated(), date_column].head(3).astype(str).tolist()
        raise ValueError(f"Wide data contains duplicate timestamps in '{date_column}': {duplicates}")
    df = df.sort_values(date_column)

    return FinancialDataset(
        values=df[targets].to_numpy(dtype=float),
        dates=df[date_column].to_numpy(),
        asset_ids=targets,
        metadata=dict(metadata or {}, source_format="wide", **time_metadata),
    )


def _load_long(
    df: pd.DataFrame,
    date_column: str,
    asset_id_column: str,
    target_column: str,
    feature_columns: Optional[List[str]],
    metadata: Optional[dict],
    freq: Optional[str],
    time_index: str,
    validate_regular: bool,
    validate_unique_index: bool,
    time_origin: str,
) -> FinancialDataset:
    required = [date_column, asset_id_column, target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required long-format columns not found: {missing}")

    df = df.copy()
    df[date_column], time_metadata = _coerce_time_column(
        df[date_column],
        date_column=date_column,
        freq=freq,
        time_index=time_index,
        validate_regular=validate_regular,
        time_origin=time_origin,
    )
    if df[asset_id_column].isna().any():
        raise ValueError(f"asset_id_column '{asset_id_column}' contains missing identifiers.")
    df[asset_id_column] = df[asset_id_column].astype(str)
    if validate_unique_index and df.duplicated([date_column, asset_id_column]).any():
        duplicates = (
            df.loc[df.duplicated([date_column, asset_id_column]), [date_column, asset_id_column]]
            .head(3)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"Long data contains duplicate ({date_column}, {asset_id_column}) rows: {duplicates}"
        )
    df = df.sort_values([date_column, asset_id_column])

    pivot = df.pivot(index=date_column, columns=asset_id_column, values=target_column)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    features = {}
    for feature in feature_columns or []:
        if feature not in df.columns:
            raise ValueError(f"Feature column '{feature}' not found in data.")
        feature_pivot = df.pivot(index=date_column, columns=asset_id_column, values=feature)
        feature_pivot = feature_pivot.reindex(index=pivot.index, columns=pivot.columns)
        features[feature] = feature_pivot.to_numpy()

    return FinancialDataset(
        values=pivot.to_numpy(dtype=float),
        dates=np.asarray(pivot.index, dtype="datetime64[ns]"),
        asset_ids=[str(col) for col in pivot.columns],
        features=features,
        metadata=dict(metadata or {}, source_format="long", **time_metadata),
    )


def load_financial_data(
    path: str,
    format: str = "wide",
    date_column: str = "date",
    target_columns: Optional[List[str]] = None,
    asset_id_column: str = "asset_id",
    target_column: str = "target",
    feature_columns: Optional[List[str]] = None,
    freq: Optional[str] = None,
    time_index: str = "auto",
    validate_regular: bool = True,
    validate_unique_index: bool = True,
    time_origin: str = "1970-01-01",
    metadata: Optional[dict] = None,
) -> FinancialDataset:
    """Load CSV/Parquet data into the canonical ``FinancialDataset`` format.

    Args:
        path: CSV or Parquet file path.
        format: Either ``"wide"`` or ``"long"``.
        date_column: Date column name.
        target_columns: Wide-format asset target columns. If omitted, all
            non-date columns are treated as targets.
        asset_id_column: Long-format asset identifier column.
        target_column: Long-format target value column.
        feature_columns: Optional long-format feature columns to pivot.
        freq: Optional expected frequency. If omitted, datetime data must have
            an inferable regular frequency when ``validate_regular`` is true.
        time_index: ``"datetime"``, ``"relative"``, or ``"auto"``. Relative
            numeric time is mapped to a frequency-aware synthetic datetime index and
            marked in metadata so models can avoid calendar semantics.
        validate_regular: Require a regular time grid.
        validate_unique_index: Reject duplicate timestamps or duplicate
            timestamp/asset pairs.
        time_origin: Origin used when mapping relative time to datetimes.
        metadata: Additional metadata attached to the dataset.
    """

    df = _read_table(path)
    normalized_format = format.lower()
    if normalized_format == "wide":
        return _load_wide(
            df,
            date_column,
            target_columns,
            metadata,
            freq=freq,
            time_index=time_index,
            validate_regular=validate_regular,
            validate_unique_index=validate_unique_index,
            time_origin=time_origin,
        )
    if normalized_format == "long":
        return _load_long(
            df,
            date_column=date_column,
            asset_id_column=asset_id_column,
            target_column=target_column,
            feature_columns=feature_columns,
            metadata=metadata,
            freq=freq,
            time_index=time_index,
            validate_regular=validate_regular,
            validate_unique_index=validate_unique_index,
            time_origin=time_origin,
        )
    raise ValueError("format must be either 'wide' or 'long'.")
