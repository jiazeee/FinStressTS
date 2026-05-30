"""DeepAR forecaster for the FinProbTS rolling-window contract.

This module implements a native PyTorch DeepAR-style model using the main
GluonTS DeepAR design choices: lagged autoregressive target inputs, LSTM
dynamics, time and age features, mean scaling, static features, Student-t
output by default, teacher-forced training, and recursive sampling.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from finprobts.data.schema import RollingWindowDataset
from finprobts.models.base import BaseProbForecastModel, ForecastResult
from finprobts.models.torch_utils import (
    relative_time_origin_and_scale,
    require_torch,
    resolve_torch_device,
    set_torch_seed,
    time_features_from_dates,
)


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def _default_lags_for_frequency(freq: Optional[str]) -> list[int]:
    if not freq:
        return [1]
    normalized = str(freq).upper()
    if normalized.startswith("M"):
        return [1, 12]
    if normalized.startswith("B"):
        return [1, 2]
    if normalized.startswith("D"):
        return [1, 7, 14]
    if normalized.startswith("H"):
        return [1, 24, 168]
    if normalized in {"T", "MIN"} or normalized.startswith("MIN"):
        return [1, 4, 12, 24, 48]
    return [1]


def _normalize_lags(lags_seq: Optional[Iterable[int]], freq: Optional[str]) -> list[int]:
    lags = list(lags_seq) if lags_seq is not None else _default_lags_for_frequency(freq)
    normalized = sorted({int(lag) for lag in lags})
    if not normalized or normalized[0] <= 0:
        raise ValueError("lags_seq must contain positive integer lags.")
    return normalized


def _auto_embedding_dims(cardinality: Sequence[int]) -> list[int]:
    return [min(50, (int(cat) + 1) // 2) for cat in cardinality]


def _age_feature(length: int) -> np.ndarray:
    return np.log10(2.0 + np.arange(int(length), dtype=np.float32)).astype(np.float32)


def _make_deepar_arrays(
    windows: RollingWindowDataset,
    use_time_features: bool,
    use_asset_static_cat: bool,
) -> Dict[str, np.ndarray]:
    past = np.asarray(windows.x_context, dtype=np.float32)
    future = np.asarray(windows.y_target, dtype=np.float32)
    past_observed = np.isfinite(past).astype(np.float32)
    future_observed = np.isfinite(future).astype(np.float32)
    past = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
    future = np.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)

    num_windows = len(windows)
    num_assets = windows.num_assets
    context_length = windows.context_length
    prediction_length = windows.prediction_length

    if use_time_features:
        origin = None
        scale = None
        if str(windows.metadata.get("time_index_kind", "datetime")).lower() == "relative":
            combined_dates = np.concatenate(
                [windows.context_dates.reshape(-1), windows.target_dates.reshape(-1)]
            )
            origin, scale = relative_time_origin_and_scale(combined_dates)
        past_time = time_features_from_dates(
            windows.context_dates,
            windows.metadata,
            origin=origin,
            scale=scale,
        )
        future_time = time_features_from_dates(
            windows.target_dates,
            windows.metadata,
            origin=origin,
            scale=scale,
        )
    else:
        past_time = np.zeros((num_windows, context_length, 0), dtype=np.float32)
        future_time = np.zeros((num_windows, prediction_length, 0), dtype=np.float32)

    age = _age_feature(context_length + prediction_length)
    past_age = np.broadcast_to(age[:context_length][None, :, None], (num_windows, context_length, 1))
    future_age = np.broadcast_to(
        age[context_length:][None, :, None],
        (num_windows, prediction_length, 1),
    )
    past_time = np.concatenate((past_time, past_age.astype(np.float32)), axis=-1)
    future_time = np.concatenate((future_time, future_age.astype(np.float32)), axis=-1)

    past_target = past.transpose(0, 2, 1).reshape(-1, context_length)
    future_target = future.transpose(0, 2, 1).reshape(-1, prediction_length)
    past_observed_values = past_observed.transpose(0, 2, 1).reshape(-1, context_length)
    future_observed_values = future_observed.transpose(0, 2, 1).reshape(-1, prediction_length)
    past_time_feat = np.repeat(past_time[:, None, :, :], num_assets, axis=1).reshape(
        -1,
        context_length,
        past_time.shape[-1],
    )
    future_time_feat = np.repeat(future_time[:, None, :, :], num_assets, axis=1).reshape(
        -1,
        prediction_length,
        future_time.shape[-1],
    )

    if use_asset_static_cat:
        static_cat = np.tile(np.arange(num_assets, dtype=np.int64), num_windows).reshape(-1, 1)
    else:
        static_cat = np.zeros((num_windows * num_assets, 1), dtype=np.int64)
    static_real = np.zeros((num_windows * num_assets, 1), dtype=np.float32)

    return {
        "past_target": past_target,
        "future_target": future_target,
        "past_observed_values": past_observed_values,
        "future_observed_values": future_observed_values,
        "past_time_feat": past_time_feat.astype(np.float32),
        "future_time_feat": future_time_feat.astype(np.float32),
        "feat_static_cat": static_cat,
        "feat_static_real": static_real,
    }


def _make_deepar_loader(
    windows: RollingWindowDataset,
    batch_size: int,
    shuffle: bool,
    use_time_features: bool,
    use_asset_static_cat: bool,
) -> Any:
    torch = require_torch()
    arrays = _make_deepar_arrays(windows, use_time_features, use_asset_static_cat)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(arrays["feat_static_cat"], dtype=torch.long),
        torch.as_tensor(arrays["feat_static_real"], dtype=torch.float32),
        torch.as_tensor(arrays["past_time_feat"], dtype=torch.float32),
        torch.as_tensor(arrays["past_target"], dtype=torch.float32),
        torch.as_tensor(arrays["past_observed_values"], dtype=torch.float32),
        torch.as_tensor(arrays["future_time_feat"], dtype=torch.float32),
        torch.as_tensor(arrays["future_target"], dtype=torch.float32),
        torch.as_tensor(arrays["future_observed_values"], dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle))


def _iter_deepar_batches(loader: Any, device: Any) -> Any:
    fields = (
        "feat_static_cat",
        "feat_static_real",
        "past_time_feat",
        "past_target",
        "past_observed_values",
        "future_time_feat",
        "future_target",
        "future_observed_values",
    )
    for batch in loader:
        yield {name: tensor.to(device) for name, tensor in zip(fields, batch)}


class DeepARNetwork(nn.Module if nn is not None else object):
    """Autoregressive univariate RNN with Student-t or Normal output."""

    def __init__(
        self,
        dynamic_feature_dim: int,
        cardinality: Sequence[int],
        embedding_dimension: Sequence[int],
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        lags_seq: Sequence[int],
        distr_output: str,
        scaling: bool,
        default_scale: Optional[float],
        min_scale: float,
        nonnegative_pred_samples: bool,
    ) -> None:
        require_torch()
        super().__init__()
        self.dynamic_feature_dim = int(dynamic_feature_dim)
        self.cardinality = [int(item) for item in cardinality]
        self.embedding_dimension = [int(item) for item in embedding_dimension]
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lags_seq = [int(lag) for lag in lags_seq]
        self.max_lag = max(self.lags_seq)
        self.distr_output = str(distr_output).lower()
        self.scaling = bool(scaling)
        self.default_scale = default_scale
        self.min_scale = float(min_scale)
        self.nonnegative_pred_samples = bool(nonnegative_pred_samples)

        if self.distr_output not in {"student_t", "studentt", "normal", "gaussian"}:
            raise ValueError("distr_output must be 'student_t' or 'normal'.")
        if len(self.cardinality) != len(self.embedding_dimension):
            raise ValueError("cardinality and embedding_dimension must have the same length.")

        self.embedders = nn.ModuleList(
            nn.Embedding(cardinality, dimension)
            for cardinality, dimension in zip(self.cardinality, self.embedding_dimension)
        )
        static_feature_dim = sum(self.embedding_dimension) + 2  # dummy static real + log scale
        input_size = len(self.lags_seq) + self.dynamic_feature_dim + static_feature_dim
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=float(dropout_rate) if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.loc_proj = nn.Linear(self.hidden_size, 1)
        self.scale_proj = nn.Linear(self.hidden_size, 1)
        self.df_proj = nn.Linear(self.hidden_size, 1) if self.distr_output in {"student_t", "studentt"} else None

    def _initial_state(self, batch_size: int, device: Any) -> tuple[Any, Any]:
        shape = (self.num_layers, int(batch_size), self.hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @staticmethod
    def _repeat_state(state: tuple[Any, Any], repeats: int) -> tuple[Any, Any]:
        return tuple(component.repeat_interleave(repeats, dim=1) for component in state)

    def _compute_scale(self, past_target: Any, past_observed: Any) -> Any:
        if not self.scaling:
            return torch.ones(
                past_target.shape[0],
                1,
                device=past_target.device,
                dtype=past_target.dtype,
            )
        observed_count = past_observed.sum(dim=1, keepdim=True)
        denom = observed_count.clamp_min(1.0)
        mean_abs = (past_target.abs() * past_observed).sum(dim=1, keepdim=True) / denom
        if self.default_scale is None:
            fallback = torch.ones_like(mean_abs)
        else:
            fallback = torch.full_like(mean_abs, float(self.default_scale))
        scale = torch.where(observed_count > 0, mean_abs.clamp_min(self.min_scale), fallback)
        return scale

    def _static_features(self, feat_static_cat: Any, feat_static_real: Any, scale: Any) -> Any:
        embeddings = [
            embedder(feat_static_cat[:, idx].clamp_min(0))
            for idx, embedder in enumerate(self.embedders)
        ]
        return torch.cat((*embeddings, feat_static_real, scale.clamp_min(self.min_scale).log()), dim=-1)

    def _features_from_indices(
        self,
        history: Any,
        step_index: int,
        time_feat: Any,
        static_feat: Any,
    ) -> Any:
        lagged = torch.stack([history[:, step_index - lag] for lag in self.lags_seq], dim=-1)
        return torch.cat((lagged, static_feat, time_feat), dim=-1)

    def _features_from_tail(self, history: Any, time_feat: Any, static_feat: Any) -> Any:
        lagged = torch.stack([history[:, -lag] for lag in self.lags_seq], dim=-1)
        return torch.cat((lagged, static_feat, time_feat), dim=-1)

    def _distribution(self, rnn_output: Any) -> Any:
        loc = self.loc_proj(rnn_output).squeeze(-1)
        scale = F.softplus(self.scale_proj(rnn_output)).squeeze(-1) + self.min_scale
        if self.df_proj is None:
            return torch.distributions.Normal(loc, scale)
        df = F.softplus(self.df_proj(rnn_output)).squeeze(-1) + 2.0
        return torch.distributions.StudentT(df=df, loc=loc, scale=scale)

    def _step(self, history: Any, time_feat: Any, static_feat: Any, state: tuple[Any, Any]) -> tuple[Any, tuple[Any, Any]]:
        rnn_input = self._features_from_tail(history, time_feat, static_feat).unsqueeze(1)
        output, next_state = self.rnn(rnn_input, state)
        return self._distribution(output[:, 0, :]), next_state

    def negative_log_likelihood(self, batch: Dict[str, Any]) -> Any:
        past = batch["past_target"]
        future = batch["future_target"]
        past_observed = batch["past_observed_values"]
        future_observed = batch["future_observed_values"]
        scale = self._compute_scale(past, past_observed)
        past_scaled = past / scale
        future_scaled = future / scale
        static_feat = self._static_features(batch["feat_static_cat"], batch["feat_static_real"], scale)
        state = self._initial_state(past.shape[0], past.device)

        weighted_losses = []
        weights = []
        for step_index in range(self.max_lag, past_scaled.shape[1]):
            features = self._features_from_indices(
                past_scaled,
                step_index=step_index,
                time_feat=batch["past_time_feat"][:, step_index, :],
                static_feat=static_feat,
            ).unsqueeze(1)
            output, state = self.rnn(features, state)
            dist = self._distribution(output[:, 0, :])
            target = past_scaled[:, step_index]
            observed = past_observed[:, step_index]
            weighted_losses.append(-dist.log_prob(target) * observed)
            weights.append(observed)

        history = past_scaled
        for step in range(future_scaled.shape[1]):
            dist, state = self._step(
                history=history,
                time_feat=batch["future_time_feat"][:, step, :],
                static_feat=static_feat,
                state=state,
            )
            target = future_scaled[:, step]
            observed = future_observed[:, step]
            weighted_losses.append(-dist.log_prob(target) * observed)
            weights.append(observed)
            history = torch.cat((history, target.unsqueeze(1)), dim=1)

        loss_sum = torch.stack(weighted_losses, dim=0).sum()
        weight_sum = torch.stack(weights, dim=0).sum()
        if float(weight_sum.detach().cpu()) <= 0.0:
            return loss_sum * 0.0
        return loss_sum / weight_sum

    def sample(self, batch: Dict[str, Any], num_samples: int) -> Any:
        past = batch["past_target"]
        past_observed = batch["past_observed_values"]
        scale = self._compute_scale(past, past_observed)
        past_scaled = past / scale
        static_feat = self._static_features(batch["feat_static_cat"], batch["feat_static_real"], scale)
        state = self._initial_state(past.shape[0], past.device)

        for step_index in range(self.max_lag, past_scaled.shape[1]):
            features = self._features_from_indices(
                past_scaled,
                step_index=step_index,
                time_feat=batch["past_time_feat"][:, step_index, :],
                static_feat=static_feat,
            ).unsqueeze(1)
            _, state = self.rnn(features, state)

        repeats = int(num_samples)
        history = past_scaled.repeat_interleave(repeats, dim=0)
        repeated_scale = scale.repeat_interleave(repeats, dim=0)
        repeated_static_feat = static_feat.repeat_interleave(repeats, dim=0)
        repeated_state = self._repeat_state(state, repeats)
        repeated_future_time_feat = batch["future_time_feat"].repeat_interleave(repeats, dim=0)

        sample_steps = []
        for step in range(repeated_future_time_feat.shape[1]):
            dist, repeated_state = self._step(
                history=history,
                time_feat=repeated_future_time_feat[:, step, :],
                static_feat=repeated_static_feat,
                state=repeated_state,
            )
            sample = dist.sample()
            sample_steps.append(sample)
            history = torch.cat((history, sample.unsqueeze(1)), dim=1)

        scaled_samples = torch.stack(sample_steps, dim=1)
        samples = scaled_samples.reshape(past.shape[0], repeats, -1) * repeated_scale.reshape(past.shape[0], repeats, 1)
        if self.nonnegative_pred_samples:
            samples = torch.relu(samples)
        return samples


class DeepARForecastModel(BaseProbForecastModel):
    """Global marginal DeepAR model.

    The model trains one shared univariate autoregressive forecaster across all
    assets. FinProbTS reassembles the marginal samples into the benchmark output
    tensor, but samples are independent across assets conditional on the shared
    model parameters.
    """

    def __init__(
        self,
        freq: Optional[str] = None,
        context_length: Optional[int] = None,
        prediction_length: Optional[int] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        lr: float = 1e-3,
        learning_rate: Optional[float] = None,
        weight_decay: float = 1e-8,
        dropout_rate: float = 0.1,
        dropout: Optional[float] = None,
        patience: Optional[int] = 10,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[Sequence[int]] = None,
        embedding_dimension: Optional[Sequence[int]] = None,
        embedding_dim: Optional[int] = None,
        distr_output: Optional[Any] = None,
        scaling: bool = True,
        default_scale: Optional[float] = None,
        lags_seq: Optional[Sequence[int]] = None,
        time_features: Optional[Any] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: Optional[int] = 50,
        imputation_method: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[Any] = None,
        validation_sampler: Optional[Any] = None,
        nonnegative_pred_samples: bool = False,
        max_epochs: int = 100,
        gradient_clip_val: Optional[float] = 10.0,
        device: str = "auto",
        seed: Optional[int] = None,
        min_scale: float = 1e-5,
        scaler_min_std: Optional[float] = None,
        verbose: bool = False,
        num_cells: Optional[int] = None,
        rnn_type: Optional[str] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        use_time_features: bool = True,
        use_asset_static_cat: bool = False,
        **_: Any,
    ) -> None:
        if num_cells is not None:
            hidden_size = int(num_cells)
        if rnn_type is not None and str(rnn_type).upper() != "LSTM":
            raise NotImplementedError("Authentic GluonTS-style DeepAR uses LSTM cells.")
        if dropout is not None:
            dropout_rate = float(dropout)
        if learning_rate is not None:
            lr = float(learning_rate)
        if optim_kwargs:
            lr = float(optim_kwargs.get("lr", lr))
            weight_decay = float(optim_kwargs.get("weight_decay", weight_decay))
            patience = optim_kwargs.get("patience", patience)
        unsupported = {
            "time_features": time_features,
            "imputation_method": imputation_method,
            "trainer_kwargs": trainer_kwargs,
            "train_sampler": train_sampler,
            "validation_sampler": validation_sampler,
        }
        unsupported = {name: value for name, value in unsupported.items() if value is not None}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                "This native FinProbTS DeepAR follows upstream defaults for these options "
                f"but does not support custom {names} yet."
            )
        if int(num_feat_dynamic_real) != 0:
            raise NotImplementedError("External dynamic real features are not wired into FinProbTS DeepAR yet.")
        if int(num_feat_static_real) != 0:
            raise NotImplementedError("External static real features are not wired into FinProbTS DeepAR yet.")

        if distr_output is None:
            distribution_name = "student_t"
        elif isinstance(distr_output, str):
            distribution_name = distr_output
        else:
            raise NotImplementedError("Custom GluonTS DistributionOutput objects are not supported yet.")

        if embedding_dim is not None:
            embedding_dimension = [int(embedding_dim)]
            num_feat_static_cat = 1
            use_asset_static_cat = True

        self.freq = None if freq is None else str(freq)
        self.context_length = None if context_length is None else int(context_length)
        self.prediction_length = None if prediction_length is None else int(prediction_length)
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.lr = float(lr)
        self.learning_rate = self.lr
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.patience = None if patience is None else int(patience)
        self.num_feat_dynamic_real = int(num_feat_dynamic_real)
        self.num_feat_static_cat = int(num_feat_static_cat)
        self.num_feat_static_real = int(num_feat_static_real)
        self.cardinality = [int(item) for item in cardinality] if cardinality is not None else None
        self.embedding_dimension = [int(item) for item in embedding_dimension] if embedding_dimension is not None else None
        self.distr_output = str(distribution_name).lower()
        self.scaling = bool(scaling)
        self.default_scale = default_scale
        self.lags_seq = _normalize_lags(lags_seq, self.freq)
        self.num_parallel_samples = int(num_parallel_samples)
        self.batch_size = int(batch_size)
        self.num_batches_per_epoch = None if num_batches_per_epoch is None else int(num_batches_per_epoch)
        self.nonnegative_pred_samples = bool(nonnegative_pred_samples)
        self.max_epochs = int(max_epochs)
        self.gradient_clip_val = None if gradient_clip_val is None else float(gradient_clip_val)
        self.device_name = device
        self.seed = seed
        self.min_scale = float(min_scale if scaler_min_std is None else scaler_min_std)
        self.verbose = bool(verbose)
        self.use_time_features = bool(use_time_features)
        self.use_asset_static_cat = bool(use_asset_static_cat)

        self._network: Optional[DeepARNetwork] = None
        self._device = None
        self._num_assets: Optional[int] = None
        self._asset_ids: Optional[list[str]] = None
        self._dynamic_feature_dim: Optional[int] = None
        self._effective_cardinality: Optional[list[int]] = None
        self._effective_embedding_dimension: Optional[list[int]] = None
        self._is_fitted = False
        self.training_history: list[Dict[str, float]] = []

    def _init_params(self) -> Dict[str, Any]:
        return {
            "freq": self.freq,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
            "patience": self.patience,
            "num_feat_dynamic_real": self.num_feat_dynamic_real,
            "num_feat_static_cat": self.num_feat_static_cat,
            "num_feat_static_real": self.num_feat_static_real,
            "cardinality": self.cardinality,
            "embedding_dimension": self.embedding_dimension,
            "distr_output": self.distr_output,
            "scaling": self.scaling,
            "default_scale": self.default_scale,
            "lags_seq": list(self.lags_seq),
            "num_parallel_samples": self.num_parallel_samples,
            "batch_size": self.batch_size,
            "num_batches_per_epoch": self.num_batches_per_epoch,
            "nonnegative_pred_samples": self.nonnegative_pred_samples,
            "max_epochs": self.max_epochs,
            "gradient_clip_val": self.gradient_clip_val,
            "device": self.device_name,
            "seed": self.seed,
            "min_scale": self.min_scale,
            "verbose": self.verbose,
            "use_time_features": self.use_time_features,
            "use_asset_static_cat": self.use_asset_static_cat,
        }

    def _validate_data(self, data: RollingWindowDataset) -> None:
        if self.context_length is not None and data.context_length != self.context_length:
            raise ValueError(
                "DeepAR context_length is controlled by the FinProbTS task window. "
                f"Got model context_length={self.context_length} but data has "
                f"context_length={data.context_length}."
            )
        if self.prediction_length is not None and data.prediction_length != self.prediction_length:
            raise ValueError(
                "DeepAR prediction_length is controlled by the FinProbTS task window. "
                f"Got model prediction_length={self.prediction_length} but data has "
                f"prediction_length={data.prediction_length}."
            )
        if data.context_length < max(self.lags_seq):
            raise ValueError(
                f"DeepAR requires context_length >= max(lags_seq). "
                f"Got context_length={data.context_length}, lags_seq={self.lags_seq}."
            )

    def _effective_static_config(self, num_assets: int) -> tuple[list[int], list[int]]:
        if self.num_feat_static_cat > 0 and not self.use_asset_static_cat:
            raise NotImplementedError(
                "Custom static categorical fields are not wired into FinProbTS DeepAR yet. "
                "Set use_asset_static_cat=True to use asset ids as the static category."
            )
        if self.use_asset_static_cat:
            cardinality = [int(num_assets)] if self.cardinality is None else list(self.cardinality)
            if cardinality[0] < int(num_assets):
                raise ValueError(
                    f"DeepAR asset static cardinality must be at least num_assets={num_assets}."
                )
        else:
            if self.cardinality not in (None, [1]):
                raise NotImplementedError(
                    "Custom static categorical cardinalities are not wired into FinProbTS DeepAR yet."
                )
            cardinality = [1]
        embedding_dimension = (
            _auto_embedding_dims(cardinality)
            if self.embedding_dimension is None
            else list(self.embedding_dimension)
        )
        if len(cardinality) != 1 or len(embedding_dimension) != 1:
            raise NotImplementedError("FinProbTS DeepAR currently supports one static categorical field.")
        return cardinality, embedding_dimension

    def _build_network(self, num_assets: int, dynamic_feature_dim: int) -> None:
        self._num_assets = int(num_assets)
        self._dynamic_feature_dim = int(dynamic_feature_dim)
        self._device = resolve_torch_device(self.device_name)
        cardinality, embedding_dimension = self._effective_static_config(num_assets)
        self._effective_cardinality = cardinality
        self._effective_embedding_dimension = embedding_dimension
        self._network = DeepARNetwork(
            dynamic_feature_dim=dynamic_feature_dim,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            distr_output=self.distr_output,
            scaling=self.scaling,
            default_scale=self.default_scale,
            min_scale=self.min_scale,
            nonnegative_pred_samples=self.nonnegative_pred_samples,
        ).to(self._device)

    def _make_loader(self, data: RollingWindowDataset, shuffle: bool) -> Any:
        return _make_deepar_loader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            use_time_features=self.use_time_features,
            use_asset_static_cat=self.use_asset_static_cat,
        )

    def _evaluate(self, loader: Any) -> float:
        assert self._network is not None and self._device is not None
        self._network.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in _iter_deepar_batches(loader, self._device):
                loss = self._network.negative_log_likelihood(batch)
                total += float(loss.detach().cpu())
                count += 1
        return total / max(count, 1)

    def fit(self, train_data: RollingWindowDataset, val_data: Optional[RollingWindowDataset] = None) -> None:
        self._validate_data(train_data)
        if val_data is not None:
            self._validate_data(val_data)
        if len(train_data) == 0:
            raise ValueError("train_data must contain at least one window.")

        require_torch()
        set_torch_seed(self.seed)
        self._asset_ids = list(train_data.asset_ids)
        sample_arrays = _make_deepar_arrays(train_data, self.use_time_features, self.use_asset_static_cat)
        self._build_network(train_data.num_assets, int(sample_arrays["past_time_feat"].shape[-1]))

        train_loader = self._make_loader(train_data, shuffle=True)
        val_loader = self._make_loader(val_data, shuffle=False) if val_data is not None and len(val_data) > 0 else None
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_score = float("inf")
        best_state = None
        bad_epochs = 0
        self.training_history = []
        for epoch in range(1, self.max_epochs + 1):
            self._network.train()
            total = 0.0
            count = 0
            for batch_index, batch in enumerate(_iter_deepar_batches(train_loader, self._device), start=1):
                optimizer.zero_grad()
                loss = self._network.negative_log_likelihood(batch)
                loss.backward()
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.gradient_clip_val)
                optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
                if self.num_batches_per_epoch is not None and batch_index >= self.num_batches_per_epoch:
                    break
            train_loss = total / max(count, 1)
            val_loss = self._evaluate(val_loader) if val_loader is not None else train_loss
            self.training_history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            if self.verbose:
                print(f"DeepAR epoch {epoch}/{self.max_epochs}: train={train_loss:.6f} val={val_loss:.6f}")
            if val_loss < best_score:
                best_score = val_loss
                best_state = copy.deepcopy(self._network.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if self.patience is not None and bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self._network.load_state_dict(best_state)
        self._is_fitted = True

    def predict(self, test_data: RollingWindowDataset, num_samples: int) -> ForecastResult:
        self._validate_data(test_data)
        if not self._is_fitted or self._network is None or self._device is None:
            raise RuntimeError("Call fit before predict.")
        if self._num_assets != test_data.num_assets:
            raise ValueError("test_data num_assets does not match the fitted model.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        loader = self._make_loader(test_data, shuffle=False)
        chunks = []
        self._network.eval()
        with torch.no_grad():
            for batch in _iter_deepar_batches(loader, self._device):
                chunks.append(self._network.sample(batch, int(num_samples)).cpu().numpy())
        flat = np.concatenate(chunks, axis=0)
        samples = flat.reshape(len(test_data), test_data.num_assets, int(num_samples), test_data.prediction_length)
        samples = samples.transpose(0, 2, 3, 1)
        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "deepar",
                "implementation": "finprobts_native_deepar",
                "reference": "GluonTS DeepAREstimator",
                "seed": self.seed,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
                "distr_output": self.distr_output,
                "lags_seq": list(self.lags_seq),
                "scaling": self.scaling,
                "num_parallel_samples_default": self.num_parallel_samples,
                "use_time_features": self.use_time_features,
                "use_asset_static_cat": self.use_asset_static_cat,
                "effective_cardinality": self._effective_cardinality,
                "effective_embedding_dimension": self._effective_embedding_dimension,
                "marginal_samples": True,
                "training_history": list(self.training_history),
            },
        )

    def save(self, path: str) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("Cannot save an unfitted DeepARForecastModel.")
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "init_params": self._init_params(),
                "model_state": self._network.state_dict(),
                "num_assets": self._num_assets,
                "asset_ids": self._asset_ids,
                "dynamic_feature_dim": self._dynamic_feature_dim,
                "effective_cardinality": self._effective_cardinality,
                "effective_embedding_dimension": self._effective_embedding_dimension,
                "training_history": self.training_history,
                "is_fitted": self._is_fitted,
            },
            output_dir / "model.pt",
        )

    @classmethod
    def load(cls, path: str) -> "DeepARForecastModel":
        require_torch()
        try:
            payload = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(Path(path) / "model.pt", map_location="cpu")
        model = cls(**payload["init_params"])
        model._asset_ids = [str(item) for item in payload.get("asset_ids", [])]
        model._build_network(int(payload["num_assets"]), int(payload.get("dynamic_feature_dim", 1)))
        model._network.load_state_dict(payload["model_state"])
        model.training_history = list(payload.get("training_history", []))
        model._is_fitted = bool(payload.get("is_fitted", True))
        return model
