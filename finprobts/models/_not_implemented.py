"""Explicit placeholders for planned native probabilistic models."""

from __future__ import annotations

from typing import Optional

from finprobts.data.schema import RollingWindowDataset
from finprobts.models.base import BaseProbForecastModel, ForecastResult


class PlannedModelPlaceholder(BaseProbForecastModel):
    """Base class for model namespaces planned after DeepVAR."""

    model_name = "planned_model"

    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError(
            f"Native `{self.model_name}` is planned but not implemented yet. "
            "The first native deep model in this milestone is `deepvar`."
        )

    def fit(self, train_data: RollingWindowDataset, val_data: Optional[RollingWindowDataset] = None) -> None:
        raise NotImplementedError

    def predict(self, test_data: RollingWindowDataset, num_samples: int) -> ForecastResult:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "PlannedModelPlaceholder":
        raise NotImplementedError
