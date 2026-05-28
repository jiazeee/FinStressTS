"""Naive probabilistic baseline model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from finprobts.data.schema import RollingWindowDataset
from finprobts.models.base import BaseProbForecastModel, ForecastResult


class NaiveForecastModel(BaseProbForecastModel):
    """Gaussian baseline estimated independently from each context window."""

    def __init__(self, seed: Optional[int] = None, min_std: float = 1e-8) -> None:
        self.seed = seed
        self.min_std = float(min_std)
        self._rng = np.random.default_rng(seed)
        self._is_fitted = False

    def fit(self, train_data: RollingWindowDataset, val_data: Optional[RollingWindowDataset] = None) -> None:
        """Mark the model fitted.

        This baseline has no trainable global parameters. It estimates
        distribution parameters locally from each context window during
        prediction.
        """

        if len(train_data) == 0:
            raise ValueError("train_data must contain at least one window.")
        self._is_fitted = True

    def predict(self, test_data: RollingWindowDataset, num_samples: int) -> ForecastResult:
        if not self._is_fitted:
            raise RuntimeError("Call fit before predict.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        x = test_data.x_context
        num_windows, _, num_assets = x.shape
        prediction_length = test_data.prediction_length

        mu = x.mean(axis=1)
        std = x.std(axis=1)
        std = np.where(std < self.min_std, self.min_std, std)

        noise = self._rng.standard_normal(
            size=(num_windows, num_samples, prediction_length, num_assets)
        )
        samples = (
            mu[:, np.newaxis, np.newaxis, :]
            + std[:, np.newaxis, np.newaxis, :] * noise
        )

        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "naive",
                "seed": self.seed,
                "min_std": self.min_std,
            },
        )

    def save(self, path: str) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": "naive",
            "seed": self.seed,
            "min_std": self.min_std,
            "is_fitted": self._is_fitted,
        }
        with open(output_dir / "model.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "NaiveForecastModel":
        with open(Path(path) / "model.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model = cls(seed=payload.get("seed"), min_std=payload.get("min_std", 1e-8))
        model._is_fitted = bool(payload.get("is_fitted", False))
        return model
