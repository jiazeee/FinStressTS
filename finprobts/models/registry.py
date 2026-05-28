"""Model registry."""

from __future__ import annotations

from typing import Any, Callable, Dict

from finprobts.models.base import BaseProbForecastModel
from finprobts.models.deepar import DeepARForecastModel
from finprobts.models.deepvar import DeepVARForecastModel
from finprobts.models.naive import NaiveForecastModel
from finprobts.models.ratd import RATDForecastModel
from finprobts.models.tempflow import TempFlowForecastModel
from finprobts.models.timegrad import TimeGradForecastModel
from finprobts.models.timemcl import TimeMCLForecastModel
from finprobts.models.tsflow import TSFlowForecastModel


ModelFactory = Callable[..., BaseProbForecastModel]


class ModelRegistry:
    """Simple name-to-factory registry for model adapters."""

    def __init__(self) -> None:
        self._factories: Dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory) -> None:
        if not name:
            raise ValueError("Model name must be non-empty.")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> BaseProbForecastModel:
        try:
            return self._factories[name](**kwargs)
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown model '{name}'. Available models: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._factories)


def get_default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("naive", NaiveForecastModel)
    registry.register("deepar", DeepARForecastModel)
    registry.register("deepvar", DeepVARForecastModel)
    registry.register("ratd", RATDForecastModel)
    registry.register("tempflow", TempFlowForecastModel)
    registry.register("timegrad", TimeGradForecastModel)
    registry.register("timemcl", TimeMCLForecastModel)
    registry.register("tsflow", TSFlowForecastModel)
    return registry
