"""CSV and Parquet loaders for financial time-series panels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

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
) -> FinancialDataset:
    if date_column not in df.columns:
        raise ValueError(f"date_column '{date_column}' not found in data.")

    targets = _resolve_target_columns(df.columns, date_column, target_columns)
    missing = [col for col in targets if col not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found: {missing}")

    df = df[[date_column] + targets].copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    return FinancialDataset(
        values=df[targets].to_numpy(dtype=float),
        dates=df[date_column].to_numpy(),
        asset_ids=targets,
        metadata=dict(metadata or {}, source_format="wide"),
    )


def _load_long(
    df: pd.DataFrame,
    date_column: str,
    asset_id_column: str,
    target_column: str,
    feature_columns: Optional[List[str]],
    metadata: Optional[dict],
) -> FinancialDataset:
    required = [date_column, asset_id_column, target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required long-format columns not found: {missing}")

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[asset_id_column] = df[asset_id_column].astype(str)
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
        metadata=dict(metadata or {}, source_format="long"),
    )


def load_financial_data(
    path: str,
    format: str = "wide",
    date_column: str = "date",
    target_columns: Optional[List[str]] = None,
    asset_id_column: str = "asset_id",
    target_column: str = "target",
    feature_columns: Optional[List[str]] = None,
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
        metadata: Additional metadata attached to the dataset.
    """

    df = _read_table(path)
    normalized_format = format.lower()
    if normalized_format == "wide":
        return _load_wide(df, date_column, target_columns, metadata)
    if normalized_format == "long":
        return _load_long(
            df,
            date_column=date_column,
            asset_id_column=asset_id_column,
            target_column=target_column,
            feature_columns=feature_columns,
            metadata=metadata,
        )
    raise ValueError("format must be either 'wide' or 'long'.")
