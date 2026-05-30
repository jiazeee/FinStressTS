from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from finprobts.cli import main
from finprobts.data import FinancialDataset, generate_rolling_windows
from finprobts.models import get_default_model_registry
from finprobts.models.deepar import DeepARForecastModel
from finprobts.models.deepvar import DeepVARForecastModel
from finprobts.models.ratd import RATDForecastModel
from finprobts.models.tempflow import TempFlowForecastModel
from finprobts.models.timegrad import TimeGradForecastModel
from finprobts.models.timemcl import TimeMCLForecastModel
from finprobts.models.tsflow import TSFlowForecastModel
from finprobts.models.torch_utils import TorchStandardScaler, make_window_arrays


def _make_dataset(num_steps: int = 48, num_assets: int = 3) -> FinancialDataset:
    t = np.arange(num_steps, dtype=float)
    values = np.stack(
        [
            0.01 * np.sin(t / 3.0 + asset) + 0.001 * asset + 0.0005 * t
            for asset in range(num_assets)
        ],
        axis=1,
    )
    return FinancialDataset(
        values=values,
        dates=pd.date_range("2024-01-01", periods=num_steps, freq="D"),
        asset_ids=[f"asset_{i}" for i in range(num_assets)],
    )


def test_make_window_arrays_masks_and_shapes():
    dataset = _make_dataset()
    values = dataset.values.copy()
    values[4, 1] = np.nan
    dataset = dataset.copy_with(values=values)
    windows = generate_rolling_windows(dataset, context_length=6, prediction_length=1)

    arrays = make_window_arrays(windows)

    assert arrays["past_target"].shape == (42, 6, 3)
    assert arrays["future_target"].shape == (42, 1, 3)
    assert arrays["past_observed_values"].shape == (42, 6, 3)
    assert arrays["target_dimension_indicator"].shape == (42, 3)
    assert arrays["past_time_feat"].shape[-1] == 4
    assert arrays["window_index"].tolist() == list(range(len(windows)))
    assert arrays["past_observed_values"][0, 4, 1] == 0.0
    assert arrays["past_target"][0, 4, 1] == 0.0


def test_torch_standard_scaler_round_trip():
    data = np.arange(24, dtype=float).reshape(2, 4, 3)
    scaler = TorchStandardScaler.fit(data)
    restored = scaler.inverse_transform_array(scaler.transform_array(data))

    assert np.allclose(restored, data)


def test_torch_standard_scaler_uses_finite_values_and_rejects_empty_dimensions():
    data = np.array(
        [
            [[1.0, np.nan], [3.0, 5.0]],
            [[np.nan, 7.0], [5.0, np.nan]],
        ]
    )

    scaler = TorchStandardScaler.fit(data)

    np.testing.assert_allclose(scaler.mean, [3.0, 6.0])
    np.testing.assert_allclose(scaler.std, [np.sqrt(8.0 / 3.0), 1.0])

    with pytest.raises(ValueError, match="no finite values"):
        TorchStandardScaler.fit(np.array([[[1.0, np.nan], [2.0, np.nan]]]))


def test_native_model_registry_names():
    names = get_default_model_registry().names()

    assert "deepar" in names
    assert "deepvar" in names
    assert "ratd" in names
    assert "tempflow" in names
    assert "timegrad" in names
    assert "timemcl" in names
    assert "tsflow" in names


def test_tsflow_residual_block_uses_native_s4_temporal_mixer():
    torch = pytest.importorskip("torch")
    from finprobts.models.tsflow.model import TSFlowResidualBlock, TSFlowS4Layer

    block = TSFlowResidualBlock(
        hidden_dim=8,
        num_features=4,
        target_dim=3,
        nheads=1,
        dropout=0.0,
        bidirectional=True,
    )

    assert isinstance(block.s4block, TSFlowS4Layer)
    assert not any(isinstance(module, torch.nn.TransformerEncoder) for module in block.modules())


def test_ratd_suppresses_exact_training_window_self_match():
    torch = pytest.importorskip("torch")
    model = RATDForecastModel()
    matrix = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.2, 0.7]], dtype=torch.float32)

    model._suppress_indexed_self_matches(matrix, torch.tensor([1, 0]), fill_value=-float("inf"))

    assert torch.isneginf(matrix[0, 1])
    assert torch.isneginf(matrix[1, 0])
    assert float(matrix[0, 0]) == pytest.approx(0.1)


def test_deepvar_smoke_fit_predict_save_load(tmp_path):
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(), context_length=8, prediction_length=1)
    train_windows = windows

    model = DeepVARForecastModel(
        hidden_size=8,
        num_layers=1,
        rank=1,
        batch_size=8,
        max_epochs=2,
        learning_rate=1.0e-2,
        seed=11,
        scaling=True,
        device="cpu",
    )
    model.fit(train_windows)
    result = model.predict(windows, num_samples=8)

    assert result.samples.shape == (40, 8, 1, 3)
    assert result.y_true.shape == (40, 1, 3)

    save_dir = tmp_path / "deepvar_model"
    model.save(str(save_dir))
    loaded = DeepVARForecastModel.load(str(save_dir))
    loaded_result = loaded.predict(windows, num_samples=4)

    assert loaded_result.samples.shape == (40, 4, 1, 3)


def test_deepvar_multistep_autoregressive_shape():
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(num_steps=56), context_length=12, prediction_length=3)
    model = DeepVARForecastModel(
        hidden_size=8,
        num_layers=1,
        rank=1,
        lags_seq=[1, 3],
        batch_size=8,
        max_epochs=1,
        learning_rate=1.0e-2,
        seed=13,
        scaling=True,
        device="cpu",
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=5)

    assert result.samples.shape == (42, 5, 3, 3)
    assert result.y_true.shape == (42, 3, 3)
    assert result.metadata["lags_seq"] == [1, 3]


def test_deepar_multistep_autoregressive_shape():
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(num_steps=56), context_length=12, prediction_length=3)
    model = DeepARForecastModel(
        hidden_size=8,
        num_layers=1,
        lags_seq=[1, 3],
        batch_size=8,
        max_epochs=1,
        learning_rate=1.0e-2,
        seed=17,
        scaling=True,
        use_asset_static_cat=True,
        device="cpu",
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=5)

    assert result.samples.shape == (42, 5, 3, 3)
    assert result.y_true.shape == (42, 3, 3)
    assert result.metadata["lags_seq"] == [1, 3]
    assert result.metadata["marginal_samples"] is True


@pytest.mark.parametrize(
    "model_cls, params",
    [
        (
            DeepARForecastModel,
            {"hidden_size": 8, "num_layers": 1, "embedding_dim": 4},
        ),
        (
            TempFlowForecastModel,
            {"num_cells": 8, "num_layers": 1, "n_blocks": 2, "hidden_size": 16, "n_hidden": 1},
        ),
        (
            TimeGradForecastModel,
            {
                "num_cells": 8,
                "num_layers": 1,
                "diff_steps": 4,
                "conditioning_length": 12,
                "residual_layers": 2,
                "residual_channels": 4,
                "residual_hidden": 16,
            },
        ),
        (
            TimeMCLForecastModel,
            {
                "num_cells": 8,
                "num_layers": 1,
                "num_hypotheses": 3,
                "mcl_hidden_dim": 16,
                "conditioning_length": 12,
                "single_linear_layer": True,
                "score_loss_weight": 0.1,
            },
        ),
        (
            RATDForecastModel,
            {
                "layers": 1,
                "channels": 8,
                "nheads": 1,
                "diffusion_steps": 4,
                "diffusion_embedding_dim": 16,
                "time_embedding_dim": 8,
                "feature_embedding_dim": 4,
                "retrieval_k": 2,
                "retrieval_metric": "cosine",
            },
        ),
        (
            TSFlowForecastModel,
            {
                "hidden_dim": 8,
                "num_residual_blocks": 1,
                "nheads": 1,
                "step_emb": 8,
                "num_steps": 4,
                "prior_context_freqs": 3,
                "use_ema": False,
            },
        ),
    ],
)
def test_next_native_models_smoke_shapes(model_cls, params):
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(), context_length=8, prediction_length=1)
    model = model_cls(
        batch_size=8,
        max_epochs=1,
        learning_rate=1.0e-2,
        seed=7,
        scaling=True,
        device="cpu",
        **params,
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=6)

    assert result.samples.shape == (40, 6, 1, 3)
    assert result.y_true.shape == (40, 1, 3)


def test_tempflow_multistep_autoregressive_shape():
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(num_steps=56), context_length=12, prediction_length=3)
    model = TempFlowForecastModel(
        num_cells=8,
        num_layers=1,
        n_blocks=2,
        hidden_size=16,
        n_hidden=1,
        conditioning_length=12,
        embedding_dimension=2,
        lags_seq=[1, 3],
        batch_size=8,
        max_epochs=1,
        lr=1.0e-2,
        seed=19,
        scaling=True,
        device="cpu",
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=5)

    assert result.samples.shape == (42, 5, 3, 3)
    assert result.y_true.shape == (42, 3, 3)
    assert result.metadata["lags_seq"] == [1, 3]
    assert result.metadata["flow_type"] == "RealNVP"


def test_timegrad_multistep_autoregressive_shape():
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(num_steps=56), context_length=12, prediction_length=3)
    model = TimeGradForecastModel(
        num_cells=8,
        num_layers=1,
        diff_steps=4,
        conditioning_length=12,
        residual_layers=2,
        residual_channels=4,
        residual_hidden=16,
        lags_seq=[1, 3],
        batch_size=8,
        max_epochs=1,
        lr=1.0e-2,
        seed=23,
        scaling=True,
        device="cpu",
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=5)

    assert result.samples.shape == (42, 5, 3, 3)
    assert result.y_true.shape == (42, 3, 3)
    assert result.metadata["lags_seq"] == [1, 3]
    assert result.metadata["diff_steps"] == 4


def test_timemcl_multistep_autoregressive_shape():
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(num_steps=56), context_length=12, prediction_length=3)
    model = TimeMCLForecastModel(
        num_cells=8,
        num_layers=1,
        num_hypotheses=4,
        mcl_hidden_dim=16,
        conditioning_length=12,
        single_linear_layer=True,
        score_loss_weight=0.1,
        lags_seq=[1, 3],
        batch_size=8,
        max_epochs=1,
        lr=1.0e-2,
        seed=29,
        scaling=True,
        device="cpu",
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=5)

    assert result.samples.shape == (42, 5, 3, 3)
    assert result.y_true.shape == (42, 3, 3)
    assert result.metadata["lags_seq"] == [1, 3]
    assert result.metadata["num_hypotheses"] == 4
    assert len(result.metadata["mean_hypothesis_scores"]) == 4


def test_ratd_multistep_reference_guided_shape():
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(num_steps=56), context_length=12, prediction_length=3)
    model = RATDForecastModel(
        layers=1,
        channels=8,
        nheads=1,
        diffusion_steps=4,
        diffusion_embedding_dim=16,
        time_embedding_dim=8,
        feature_embedding_dim=4,
        retrieval_k=2,
        retrieval_metric="cosine",
        batch_size=8,
        max_epochs=1,
        learning_rate=1.0e-2,
        seed=31,
        scaling=True,
        device="cpu",
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=5)

    assert result.samples.shape == (42, 5, 3, 3)
    assert result.y_true.shape == (42, 3, 3)
    assert result.metadata["retrieval_k"] == 2
    assert result.metadata["use_reference"] is True


def test_tsflow_multistep_gp_prior_shape():
    pytest.importorskip("torch")
    windows = generate_rolling_windows(_make_dataset(num_steps=56), context_length=12, prediction_length=3)
    model = TSFlowForecastModel(
        hidden_dim=8,
        num_residual_blocks=1,
        nheads=1,
        step_emb=8,
        num_steps=4,
        prior_context_freqs=2,
        use_lags=True,
        lags_seq=[1, 3],
        use_ema=False,
        batch_size=8,
        max_epochs=1,
        learning_rate=1.0e-2,
        seed=37,
        scaling=True,
        device="cpu",
    )

    model.fit(windows)
    result = model.predict(windows, num_samples=5)

    assert result.samples.shape == (42, 5, 3, 3)
    assert result.y_true.shape == (42, 3, 3)
    assert result.metadata["lags_seq"] == [1, 3]
    assert result.metadata["prior_kernel"] == "ou"
    assert result.metadata["s4_backend"] == "native_diagonal_state_space"


def test_cli_deepvar_smoke_run(tmp_path):
    pytest.importorskip("torch")
    data_path = tmp_path / "returns.csv"
    dataset = _make_dataset(num_steps=60, num_assets=3)
    pd.DataFrame(
        {
            "date": dataset.dates.astype("datetime64[ns]").astype(str),
            "a": dataset.values[:, 0],
            "b": dataset.values[:, 1],
            "c": dataset.values[:, 2],
        }
    ).to_csv(data_path, index=False)

    output_dir = tmp_path / "outputs"
    config = {
        "run": {"run_id": "deepvar_smoke", "output_dir": str(output_dir), "seed": 123},
        "dataset": {
            "name": "custom_csv",
            "path": str(data_path),
            "format": "wide",
            "date_column": "date",
        },
        "preprocessing": {
            "value_kind": "returns",
            "missing_method": "ffill",
            "standardize": True,
        },
        "split": {"train_size": 0.6, "val_size": 0.2},
        "task": {"context_length": 8, "prediction_length": 1, "stride": 1},
        "model": {
            "name": "deepvar",
            "params": {
                "hidden_size": 8,
                "num_layers": 1,
                "rank": 1,
                "batch_size": 8,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "seed": 123,
                "device": "cpu",
            },
        },
        "forecast": {"num_samples": 5},
    }
    config_path = tmp_path / "deepvar_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == 0
    assert (output_dir / "deepvar_smoke" / "forecast_samples.npz").exists()
    assert (output_dir / "deepvar_smoke" / "forecast_metrics.json").exists()
