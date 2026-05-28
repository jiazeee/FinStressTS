from __future__ import annotations

import numpy as np
import pandas as pd

from finprobts.data import (
    FinancialDataset,
    generate_rolling_windows,
    handle_missing_values,
    load_financial_data,
    price_to_log_return,
    time_train_val_test_split,
)


def test_wide_csv_loading(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "BTC": [100.0, 101.0, 103.0],
            "ETH": [50.0, 51.0, 52.0],
        }
    ).to_csv(path, index=False)

    dataset = load_financial_data(str(path), format="wide", date_column="date")

    assert dataset.values.shape == (3, 2)
    assert dataset.asset_ids == ["BTC", "ETH"]


def test_long_csv_loading(tmp_path):
    path = tmp_path / "long.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "asset_id": ["BTC", "ETH", "BTC", "ETH"],
            "target": [0.01, 0.02, 0.03, 0.04],
            "volume": [10, 20, 11, 21],
        }
    ).to_csv(path, index=False)

    dataset = load_financial_data(
        str(path),
        format="long",
        date_column="date",
        asset_id_column="asset_id",
        target_column="target",
        feature_columns=["volume"],
    )

    assert dataset.values.shape == (2, 2)
    assert dataset.asset_ids == ["BTC", "ETH"]
    assert dataset.features["volume"].shape == (2, 2)


def test_log_return_transformation():
    prices = np.array([[100.0, 50.0], [110.0, 55.0], [121.0, 60.5]])
    returns = price_to_log_return(prices)

    assert returns.shape == (2, 2)
    np.testing.assert_allclose(returns[:, 0], [np.log(1.1), np.log(1.1)])


def test_missing_value_handling():
    dataset = FinancialDataset(
        values=np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 5.0]]),
        dates=pd.date_range("2024-01-01", periods=3),
        asset_ids=["a", "b"],
    )

    filled = handle_missing_values(dataset, method="ffill")

    assert not np.isnan(filled.values).any()
    assert filled.values[0, 1] == 3.0
    assert filled.values[2, 0] == 2.0


def test_chronological_split():
    dataset = FinancialDataset(
        values=np.arange(20, dtype=float).reshape(10, 2),
        dates=pd.date_range("2024-01-01", periods=10),
        asset_ids=["a", "b"],
    )

    split = time_train_val_test_split(dataset, train_size=0.5, val_size=0.3)

    assert split.train.num_timesteps == 5
    assert split.val.num_timesteps == 3
    assert split.test.num_timesteps == 2
    assert split.train.dates[-1] < split.val.dates[0] < split.test.dates[0]


def test_rolling_window_generation():
    dataset = FinancialDataset(
        values=np.arange(20, dtype=float).reshape(10, 2),
        dates=pd.date_range("2024-01-01", periods=10),
        asset_ids=["a", "b"],
    )

    windows = generate_rolling_windows(dataset, context_length=3, prediction_length=2, stride=1)

    assert windows.x_context.shape == (6, 3, 2)
    assert windows.y_target.shape == (6, 2, 2)
    np.testing.assert_array_equal(windows.x_context[0], dataset.values[:3])
    np.testing.assert_array_equal(windows.y_target[0], dataset.values[3:5])
