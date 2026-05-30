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
            "time_feature_origin": str(dataset.dates[0]) if dataset.num_timesteps else None,
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


def generate_boundary_aware_rolling_windows(
    history_dataset: FinancialDataset,
    target_dataset: FinancialDataset,
    context_length: int,
    prediction_length: int,
    stride: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> RollingWindowDataset:
    """Generate eval windows whose targets stay inside ``target_dataset``.

    The context may use the tail of ``history_dataset`` plus already observed
    points from ``target_dataset``. This matches rolling-origin forecasting:
    validation/test labels are not leaked into earlier targets, but forecasts
    are allowed to condition on observations that would be known by that origin.
    """

    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if list(history_dataset.asset_ids) != list(target_dataset.asset_ids):
        raise ValueError("history_dataset and target_dataset must have identical asset_ids.")
    if target_dataset.num_timesteps < prediction_length:
        raise ValueError("target_dataset is too short for the requested prediction_length.")

    history_length = min(history_dataset.num_timesteps, int(context_length))
    history_start = history_dataset.num_timesteps - history_length
    history_values = history_dataset.values[history_start:]
    history_dates = history_dataset.dates[history_start:]

    values = np.concatenate([history_values, target_dataset.values], axis=0)
    dates = np.concatenate([history_dates, target_dataset.dates], axis=0)
    boundary = history_length
    total_length = values.shape[0]
    if total_length < context_length + prediction_length:
        raise ValueError(
            "History plus target data is too short for boundary-aware windows: "
            f"{total_length} < {context_length + prediction_length}."
        )

    x_windows = []
    y_windows = []
    context_dates = []
    target_dates = []
    first_target_start = max(int(context_length), boundary)
    last_target_start = total_length - int(prediction_length)
    for target_start in range(first_target_start, last_target_start + 1, int(stride)):
        target_end = target_start + int(prediction_length)
        if target_start < boundary or target_end > total_length:
            continue
        context_start = target_start - int(context_length)
        x_windows.append(values[context_start:target_start])
        y_windows.append(values[target_start:target_end])
        context_dates.append(dates[context_start:target_start])
        target_dates.append(dates[target_start:target_end])

    if not x_windows:
        raise ValueError("No boundary-aware rolling windows could be generated.")

    task_metadata = dict(target_dataset.metadata)
    task_metadata.update(metadata or {})
    task_metadata.update(
        {
            "context_length": int(context_length),
            "prediction_length": int(prediction_length),
            "stride": int(stride),
            "boundary_aware": True,
            "history_length_used": int(history_length),
            "target_start_boundary": int(boundary),
            "time_feature_origin": str(dates[0]) if len(dates) else None,
        }
    )

    return RollingWindowDataset(
        x_context=np.stack(x_windows, axis=0),
        y_target=np.stack(y_windows, axis=0),
        context_dates=np.stack(context_dates, axis=0),
        target_dates=np.stack(target_dates, axis=0),
        asset_ids=list(target_dataset.asset_ids),
        metadata=task_metadata,
    )
