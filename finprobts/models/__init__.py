"""Model interfaces and built-in benchmark baselines."""

from finprobts.models.base import BaseProbForecastModel, ForecastResult
from finprobts.models.deepar import DeepARForecastModel
from finprobts.models.deepvar import DeepVARForecastModel
from finprobts.models.naive import NaiveForecastModel
from finprobts.models.ratd import RATDForecastModel
from finprobts.models.registry import ModelRegistry, get_default_model_registry
from finprobts.models.tempflow import TempFlowForecastModel
from finprobts.models.timegrad import TimeGradForecastModel
from finprobts.models.timemcl import TimeMCLForecastModel
from finprobts.models.tsflow import TSFlowForecastModel

__all__ = [
    "BaseProbForecastModel",
    "DeepARForecastModel",
    "DeepVARForecastModel",
    "ForecastResult",
    "ModelRegistry",
    "NaiveForecastModel",
    "RATDForecastModel",
    "TempFlowForecastModel",
    "TimeGradForecastModel",
    "TimeMCLForecastModel",
    "TSFlowForecastModel",
    "get_default_model_registry",
]
