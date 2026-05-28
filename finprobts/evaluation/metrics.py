"""Point, probabilistic, and finance-specific forecast metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from finprobts.models.base import ForecastResult


def _samples_and_target(result: ForecastResult) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(result.samples, dtype=float), np.asarray(result.y_true, dtype=float)


def _point_forecast(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1)


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def nmae_sigma(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    """Volatility-normalized MAE using pooled target standard deviation.

    This is the paper-style point metric: mean absolute error divided by the
    empirical standard deviation of all realized test values across windows,
    horizons, and assets.
    """

    pooled_std = float(np.std(np.asarray(y_true, dtype=float)))
    return mae(y_pred, y_true) / max(pooled_std, eps)


def nd(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sum(np.abs(y_pred - y_true)) / max(np.sum(np.abs(y_true)), eps))


def quantile_loss(samples: np.ndarray, y_true: np.ndarray, quantile: float) -> float:
    """Mean pinball loss for a forecast quantile."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1).")
    q_hat = np.quantile(samples, quantile, axis=1)
    error = y_true - q_hat
    loss = np.where(error >= 0, quantile * error, (quantile - 1.0) * error)
    return float(np.mean(loss))


def empirical_coverage(samples: np.ndarray, y_true: np.ndarray, level: float) -> float:
    """Empirical coverage of central prediction intervals."""

    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1).")
    alpha = (1.0 - level) / 2.0
    lower = np.quantile(samples, alpha, axis=1)
    upper = np.quantile(samples, 1.0 - alpha, axis=1)
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def crps_sample(samples: np.ndarray, y_true: np.ndarray) -> float:
    """Approximate CRPS from empirical samples.

    Args:
        samples: ``[num_windows, num_samples, prediction_length, num_assets]``.
        y_true: ``[num_windows, prediction_length, num_assets]``.
    """

    samples = np.asarray(samples, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    sample_count = samples.shape[1]
    if sample_count < 2:
        raise ValueError("CRPS requires at least two forecast samples.")

    term1 = np.abs(samples - y_true[:, np.newaxis, :, :]).mean(axis=1)
    sorted_samples = np.sort(samples, axis=1)
    weights = (2 * np.arange(1, sample_count + 1) - sample_count - 1) / (sample_count ** 2)
    term2 = (sorted_samples * weights.reshape(1, sample_count, 1, 1)).sum(axis=1)
    return float((term1 - term2).mean())


def crps_sum_sample(samples: np.ndarray, y_true: np.ndarray) -> float:
    """Approximate CRPS for the cross-asset sum distribution."""

    summed_samples = samples.sum(axis=-1)[:, :, :, np.newaxis]
    summed_true = y_true.sum(axis=-1)[:, :, np.newaxis]
    return crps_sample(summed_samples, summed_true)


def crps_sum_normalized_sample(samples: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Scale-normalized CRPS for the cross-asset sum distribution."""

    summed_true = np.asarray(y_true, dtype=float).sum(axis=-1)
    scale = float(np.mean(np.abs(summed_true)))
    return crps_sum_sample(samples, y_true) / max(scale, eps)


def var_violation_rate(samples: np.ndarray, y_true: np.ndarray, alpha: float = 0.05) -> float:
    """Rate at which realized returns fall below sample-estimated VaR."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    var = np.quantile(samples, alpha, axis=1)
    return float((y_true < var).mean())


def expected_shortfall_estimate(samples: np.ndarray, alpha: float = 0.05) -> float:
    """Average left-tail expected shortfall estimated from samples."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    var = np.quantile(samples, alpha, axis=1)
    tail = np.where(samples <= var[:, np.newaxis, :, :], samples, np.nan)
    with np.errstate(invalid="ignore"):
        es = np.nanmean(tail, axis=1)
    return float(np.nanmean(es))


def volatility_forecast_error(samples: np.ndarray, y_true: np.ndarray) -> float:
    """MAE between sample forecast volatility and an absolute-error volatility proxy."""

    median = np.quantile(samples, 0.5, axis=1)
    forecast_vol = samples.std(axis=1)
    realized_proxy = np.abs(y_true - median)
    return float(np.mean(np.abs(forecast_vol - realized_proxy)))


def correlation_forecast_error(samples: np.ndarray, y_true: np.ndarray) -> Optional[float]:
    """Average Frobenius error between predicted and realized asset correlations."""

    _, _, prediction_length, num_assets = samples.shape
    if num_assets < 2 or prediction_length < 2:
        return None

    errors = []
    for window_idx in range(samples.shape[0]):
        predicted_panel = samples[window_idx].reshape(-1, num_assets)
        realized_panel = y_true[window_idx]
        pred_corr = np.corrcoef(predicted_panel, rowvar=False)
        true_corr = np.corrcoef(realized_panel, rowvar=False)
        if np.isnan(pred_corr).any() or np.isnan(true_corr).any():
            continue
        errors.append(np.linalg.norm(pred_corr - true_corr, ord="fro"))

    if not errors:
        return None
    return float(np.mean(errors))


def evaluate_forecasts(
    result: ForecastResult,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    coverage_levels: Iterable[float] = (0.5, 0.9),
    var_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate all MVP forecast metrics and return JSON-serializable values."""

    samples, y_true = _samples_and_target(result)
    point = _point_forecast(samples)

    metrics: Dict[str, Any] = {
        "mae": mae(point, y_true),
        "nmae_sigma": nmae_sigma(point, y_true),
        "rmse": rmse(point, y_true),
        "mape": mape(point, y_true),
        "nd": nd(point, y_true),
        "crps": crps_sample(samples, y_true),
        "crps_sum": crps_sum_sample(samples, y_true),
        "crps_sum_normalized": crps_sum_normalized_sample(samples, y_true),
        "var_violation_rate": var_violation_rate(samples, y_true, alpha=var_alpha),
        "expected_shortfall": expected_shortfall_estimate(samples, alpha=var_alpha),
        "volatility_forecast_error": volatility_forecast_error(samples, y_true),
        "correlation_forecast_error": correlation_forecast_error(samples, y_true),
    }

    for q in quantiles:
        metrics[f"quantile_loss_{q:.2f}"] = quantile_loss(samples, y_true, q)
    for level in coverage_levels:
        metrics[f"coverage_{level:.2f}"] = empirical_coverage(samples, y_true, level)
    return metrics
