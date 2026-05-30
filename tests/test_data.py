from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finprobts.data import (
    DatasetNormalizer,
    FinancialDataset,
    TimeSeriesSplit,
    generate_boundary_aware_rolling_windows,
    generate_rolling_windows,
    handle_missing_values,
    handle_missing_values_split_safe,
    load_financial_data,
    price_to_log_return,
    time_train_val_test_split,
)
from finprobts.models.torch_utils import make_window_arrays


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
        values=np.array([[1.0, 2.0], [2.0, 3.0], [np.nan, 5.0]]),
        dates=pd.date_range("2024-01-01", periods=3),
        asset_ids=["a", "b"],
    )

    filled = handle_missing_values(dataset, method="ffill")

    assert not np.isnan(filled.values).any()
    assert filled.values[2, 0] == 2.0


def test_ffill_is_causal_and_requires_initial_for_leading_gaps():
    dataset = FinancialDataset(
        values=np.array([[np.nan, 2.0], [2.0, 3.0]]),
        dates=pd.date_range("2024-01-01", periods=2),
        asset_ids=["a", "b"],
    )

    with pytest.raises(ValueError, match="causal ffill"):
        handle_missing_values(dataset, method="ffill")

    filled = handle_missing_values(dataset, method="ffill", initial_values=np.array([9.0, 8.0]))

    assert filled.values[0, 0] == 9.0
    assert filled.values[0, 1] == 2.0


def test_split_safe_ffill_does_not_backfill_from_future_split():
    dataset = FinancialDataset(
        values=np.array([[1.0], [2.0], [3.0], [np.nan], [5.0]]),
        dates=pd.date_range("2024-01-01", periods=5),
        asset_ids=["a"],
    )
    split = TimeSeriesSplit(
        train=dataset.slice_time(0, 3),
        val=dataset.slice_time(3, 4),
        test=dataset.slice_time(4, 5),
    )

    filled = handle_missing_values_split_safe(split, method="ffill")

    assert filled.val.values[0, 0] == 3.0
    assert filled.test.values[0, 0] == 5.0


def test_dataset_normalizer_uses_finite_train_values_and_rejects_empty_assets():
    dataset = FinancialDataset(
        values=np.array([[1.0, np.nan], [3.0, 5.0], [np.nan, 7.0]]),
        dates=pd.date_range("2024-01-01", periods=3),
        asset_ids=["a", "b"],
    )

    normalizer = DatasetNormalizer.fit(dataset)

    np.testing.assert_allclose(normalizer.mean, [2.0, 6.0])
    np.testing.assert_allclose(normalizer.std, [1.0, 1.0])

    all_missing = FinancialDataset(
        values=np.array([[1.0, np.nan], [2.0, np.nan]]),
        dates=pd.date_range("2024-01-01", periods=2),
        asset_ids=["a", "b"],
    )
    with pytest.raises(ValueError, match="no finite training values"):
        DatasetNormalizer.fit(all_missing)


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


def test_boundary_aware_windows_use_prior_history_for_eval_targets():
    dataset = FinancialDataset(
        values=np.arange(10, dtype=float).reshape(10, 1),
        dates=pd.date_range("2024-01-01", periods=10),
        asset_ids=["a"],
    )
    train = dataset.slice_time(0, 6)
    val = dataset.slice_time(6, 8)

    windows = generate_boundary_aware_rolling_windows(train, val, context_length=3, prediction_length=1)

    assert windows.x_context.shape == (2, 3, 1)
    np.testing.assert_array_equal(windows.x_context[0, :, 0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(windows.y_target[:, 0, 0], [6.0, 7.0])


def test_numeric_relative_time_is_not_treated_as_nanoseconds(tmp_path):
    path = tmp_path / "long.csv"
    pd.DataFrame(
        {
            "time": [0, 0, 1, 1, 2, 2],
            "asset_id": ["a", "b", "a", "b", "a", "b"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(path, index=False)

    dataset = load_financial_data(
        str(path),
        format="long",
        date_column="time",
        asset_id_column="asset_id",
        target_column="target",
        time_index="relative",
        freq="D",
    )

    assert dataset.metadata["time_index_kind"] == "relative"
    assert dataset.metadata["freq"] == "D"
    assert (dataset.dates[1] - dataset.dates[0]) == np.timedelta64(1, "D")


def test_relative_window_time_features_use_shared_scale():
    dataset = FinancialDataset(
        values=np.arange(6, dtype=float).reshape(6, 1),
        dates=np.arange(6),
        asset_ids=["a"],
        metadata={"time_index_kind": "relative", "freq": "D"},
    )
    windows = generate_rolling_windows(dataset, context_length=3, prediction_length=1)

    arrays = make_window_arrays(windows)

    date_three_as_future = arrays["future_time_feat"][0, 0, 0]
    date_three_as_context = arrays["past_time_feat"][1, 2, 0]
    assert date_three_as_future == date_three_as_context


def test_financial_dataset_numeric_relative_dates_preserve_gaps():
    dataset = FinancialDataset(
        values=np.arange(3, dtype=float).reshape(3, 1),
        dates=np.array([0, 2, 5]),
        asset_ids=["a"],
        metadata={"time_index_kind": "relative", "freq": "D"},
    )

    assert (dataset.dates[1] - dataset.dates[0]) == np.timedelta64(2, "D")
    assert (dataset.dates[2] - dataset.dates[1]) == np.timedelta64(3, "D")


def test_long_loader_rejects_missing_asset_ids_before_string_cast(tmp_path):
    path = tmp_path / "missing_asset.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "asset_id": ["a", np.nan, "a"],
            "target": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing identifiers"):
        load_financial_data(
            str(path),
            format="long",
            date_column="date",
            asset_id_column="asset_id",
            target_column="target",
            validate_regular=False,
        )


def test_relative_time_metadata_reflects_disabled_regular_validation(tmp_path):
    path = tmp_path / "relative_irregular.csv"
    pd.DataFrame(
        {
            "time": [0, 0, 2, 2, 5, 5],
            "asset_id": ["a", "b", "a", "b", "a", "b"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(path, index=False)

    dataset = load_financial_data(
        str(path),
        format="long",
        date_column="time",
        asset_id_column="asset_id",
        target_column="target",
        time_index="relative",
        validate_regular=False,
    )

    assert dataset.metadata["time_index_kind"] == "relative"
    assert dataset.metadata["time_index_is_regular"] is False
    assert (dataset.dates[1] - dataset.dates[0]) == np.timedelta64(2, "D")
    assert (dataset.dates[2] - dataset.dates[1]) == np.timedelta64(3, "D")


def test_duplicate_and_irregular_time_validation(tmp_path):
    duplicate_path = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "a": [1.0, 2.0, 3.0],
        }
    ).to_csv(duplicate_path, index=False)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_financial_data(str(duplicate_path), format="wide", date_column="date")

    irregular_path = tmp_path / "irregular.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-04"],
            "a": [1.0, 2.0, 3.0],
        }
    ).to_csv(irregular_path, index=False)

    with pytest.raises(ValueError, match="regular frequency"):
        load_financial_data(str(irregular_path), format="wide", date_column="date")
