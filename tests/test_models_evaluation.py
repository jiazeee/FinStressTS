from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finprobts.data import FinancialDataset, generate_rolling_windows
from finprobts.evaluation import crps_sample, crps_sum_normalized_sample, crps_sum_sample
from finprobts.models import ForecastResult, NaiveForecastModel


def test_forecast_result_shape_validation():
    samples = np.zeros((2, 5, 3, 4))
    y_true = np.zeros((2, 3, 4))
    result = ForecastResult(
        samples=samples,
        y_true=y_true,
        start_dates=pd.date_range("2024-01-01", periods=2),
        item_ids=["a", "b", "c", "d"],
    )

    assert result.num_windows == 2

    with pytest.raises(ValueError):
        ForecastResult(
            samples=np.zeros((2, 5, 3, 4)),
            y_true=np.zeros((2, 4, 4)),
            start_dates=pd.date_range("2024-01-01", periods=2),
            item_ids=["a", "b", "c", "d"],
        )


def test_naive_model_forecast_sample_shape():
    dataset = FinancialDataset(
        values=np.arange(60, dtype=float).reshape(20, 3),
        dates=pd.date_range("2024-01-01", periods=20),
        asset_ids=["a", "b", "c"],
    )
    windows = generate_rolling_windows(dataset, context_length=5, prediction_length=2)

    model = NaiveForecastModel(seed=1)
    model.fit(windows)
    result = model.predict(windows, num_samples=7)

    assert result.samples.shape == (14, 7, 2, 3)
    assert result.y_true.shape == (14, 2, 3)


def test_crps_sample_approximation_perfect_samples():
    y_true = np.array([[[1.0], [2.0]]])
    samples = np.repeat(y_true[:, np.newaxis, :, :], repeats=3, axis=1)

    assert crps_sample(samples, y_true) == pytest.approx(0.0)


def _pairwise_crps_1d(samples: np.ndarray, y: float) -> float:
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    term1 = np.mean(np.abs(x - float(y)))
    term2 = np.mean(np.abs(x[:, None] - x[None, :]))
    return float(term1 - 0.5 * term2)


def test_crps_sum_matches_pairwise_paper_estimator():
    samples = np.array(
        [
            [
                [[1.0, 2.0], [1.5, 2.5]],
                [[2.0, 1.0], [2.5, 1.5]],
                [[3.0, 4.0], [3.5, 4.5]],
            ],
            [
                [[0.0, 1.0], [0.5, 1.5]],
                [[1.0, 0.0], [1.5, 0.5]],
                [[2.0, 3.0], [2.5, 3.5]],
            ],
        ]
    )
    y_true = np.array(
        [
            [[2.0, 1.0], [2.0, 2.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )

    summed_samples = samples.sum(axis=-1)
    summed_true = y_true.sum(axis=-1)
    expected = np.mean(
        [
            _pairwise_crps_1d(summed_samples[w, :, h], summed_true[w, h])
            for w in range(samples.shape[0])
            for h in range(samples.shape[2])
        ]
    )

    assert crps_sum_sample(samples, y_true) == pytest.approx(expected)


def test_crps_sum_normalized_uses_mean_absolute_sum_target_scale():
    samples = np.array([[[[1.0, 2.0]], [[2.0, 1.0]], [[3.0, 4.0]]]])
    y_true = np.array([[[2.0, 1.0]]])

    raw = crps_sum_sample(samples, y_true)
    scale = np.mean(np.abs(y_true.sum(axis=-1)))

    assert crps_sum_normalized_sample(samples, y_true) == pytest.approx(raw / scale)
