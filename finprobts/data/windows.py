"""Rolling-window task generation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from finprobts.data.schema import FinancialDataset, RollingWindowDataset


def generate_rolling_windows(
    dataset: FinancialDataset,
    context_length: int,
    prediction_length: int,
    stride: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> RollingWindowDataset:
    """Generate rolling context/target windows from a financial dataset."""

    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    total_length = context_length + prediction_length
    if dataset.num_timesteps < total_length:
        raise ValueError(
            "Dataset is too short for the requested window lengths: "
            f"{dataset.num_timesteps} < {total_length}."
        )

    x_windows = []
    y_windows = []
    context_dates = []
    target_dates = []

    last_start = dataset.num_timesteps - total_length
    for start in range(0, last_start + 1, stride):
        context_end = start + context_length
        target_end = context_end + prediction_length
        x_windows.append(dataset.values[start:context_end])
        y_windows.append(dataset.values[context_end:target_end])
        context_dates.append(dataset.dates[start:context_end])
        target_dates.append(dataset.dates[context_end:target_end])

    task_metadata = dict(dataset.metadata)
    task_metadata.update(metadata or {})
    task_metadata.update(
        {
            "context_length": int(context_length),
            "prediction_length": int(prediction_length),
            "stride": int(stride),
        }
    )

    return RollingWindowDataset(
        x_context=np.stack(x_windows, axis=0),
        y_target=np.stack(y_windows, axis=0),
        context_dates=np.stack(context_dates, axis=0),
        target_dates=np.stack(target_dates, axis=0),
        asset_ids=list(dataset.asset_ids),
        metadata=task_metadata,
    )
