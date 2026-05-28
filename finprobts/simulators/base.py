"""
Base simulator class for all time series simulators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class BaseSimulator:
    """Base class for all time series simulators with local RNG for reproducibility."""

    def __init__(self, seed: Optional[int] = None, burn_in: int = 200):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.burn_in = int(burn_in)
        self._simulation_result = None

    def simulate(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement simulate()")

    def to_dataframe(self) -> pd.DataFrame:
        if self._simulation_result is None:
            raise ValueError("No simulation has been run yet. Call simulate() first.")
        return self._convert_to_dataframe(self._simulation_result)

    def _convert_to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _convert_to_dataframe()")

    def get_params(self) -> Dict[str, Any]:
        if self._simulation_result is None:
            raise ValueError("No simulation has been run yet. Call simulate() first.")
        return self._simulation_result.get("params", {})
