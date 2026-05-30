"""Data loading, preprocessing, and task generation utilities."""

from finprobts.data.io import load_financial_data
from finprobts.data.preprocessing import (
    DatasetNormalizer,
    handle_missing_values,
    handle_missing_values_split_safe,
    price_to_log_return,
    rolling_normalize,
    time_train_val_test_split,
)
from finprobts.data.registry import DatasetRegistry, get_default_dataset_registry
from finprobts.data.schema import FinancialDataset, RollingWindowDataset, TimeSeriesSplit, concatenate_financial_datasets
from finprobts.data.windows import generate_boundary_aware_rolling_windows, generate_rolling_windows

__all__ = [
    "DatasetNormalizer",
    "DatasetRegistry",
    "FinancialDataset",
    "RollingWindowDataset",
    "TimeSeriesSplit",
    "concatenate_financial_datasets",
    "generate_boundary_aware_rolling_windows",
    "generate_rolling_windows",
    "get_default_dataset_registry",
    "handle_missing_values",
    "handle_missing_values_split_safe",
    "load_financial_data",
    "price_to_log_return",
    "rolling_normalize",
    "time_train_val_test_split",
]
