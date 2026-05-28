from __future__ import annotations

import matplotlib

from finprobts.experiment.data_efficiency import _plot_crps_data_efficiency


def test_data_efficiency_plot_skips_failed_and_missing_crps_rows(tmp_path):
    rows = [
        {
            "dataset_id": "dataset_a",
            "model_id": "naive",
            "train_fraction": 0.5,
            "crps": 0.2,
            "exit_code": 0,
        },
        {
            "dataset_id": "dataset_a",
            "model_id": "naive",
            "train_fraction": 1.0,
            "crps": 0.1,
            "exit_code": 0,
        },
        {
            "dataset_id": "dataset_b",
            "model_id": "naive",
            "train_fraction": 0.5,
            "crps": 0.9,
            "exit_code": 1,
        },
        {
            "dataset_id": "dataset_c",
            "model_id": "naive",
            "train_fraction": 0.5,
            "exit_code": 0,
        },
    ]

    plot_paths = _plot_crps_data_efficiency(rows, tmp_path)

    assert matplotlib.get_backend().lower() == "agg"
    assert len(plot_paths) == 1
    assert plot_paths[0].name == "data_efficiency_crps_naive.png"
    assert plot_paths[0].exists()
    assert plot_paths[0].stat().st_size > 0
