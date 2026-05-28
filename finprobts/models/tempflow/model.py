"""TempFlow forecaster for the FinProbTS rolling-window contract.

This module implements a native PyTorch TempFlow-style model following the
PyTorchTS design: lagged multivariate autoregressive inputs, recurrent temporal
conditioning, target-dimension embeddings, mean scaling, and a conditional
normalizing flow over the target vector.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from finprobts.data.schema import RollingWindowDataset
from finprobts.models.base import BaseProbForecastModel, ForecastResult
from finprobts.models.torch_utils import (
    iter_torch_batches,
    make_torch_data_loader,
    require_torch,
    resolve_torch_device,
    set_torch_seed,
)


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


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


class ConditionalAffineCoupling(nn.Module if nn is not None else object):
    """RealNVP affine coupling layer conditioned on RNN-projected state."""

    def __init__(self, dim: int, cond_dim: int, hidden_size: int, n_hidden: int, mask: Any) -> None:
        require_torch()
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("mask", mask.reshape(1, self.dim))
        layers = []
        input_dim = self.dim + int(cond_dim)
        depth = max(1, int(n_hidden))
        for idx in range(depth):
            layers.append(nn.Linear(input_dim if idx == 0 else int(hidden_size), int(hidden_size)))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(int(hidden_size), self.dim * 2))
        self.net = nn.Sequential(*layers)

    def _shift_log_scale(self, x: Any, cond: Any) -> tuple[Any, Any]:
        masked_x = x * self.mask
        params = self.net(torch.cat((masked_x, cond), dim=-1))
        shift, log_scale = params.chunk(2, dim=-1)
        inv_mask = 1.0 - self.mask
        return shift * inv_mask, torch.tanh(log_scale) * inv_mask

    def forward(self, x: Any, cond: Any) -> tuple[Any, Any]:
        shift, log_scale = self._shift_log_scale(x, cond)
        inv_mask = 1.0 - self.mask
        z = x * self.mask + inv_mask * ((x - shift) * torch.exp(-log_scale))
        log_det = -(log_scale * inv_mask).sum(dim=-1)
        return z, log_det

    def inverse(self, z: Any, cond: Any) -> Any:
        shift, log_scale = self._shift_log_scale(z, cond)
        inv_mask = 1.0 - self.mask
        return z * self.mask + inv_mask * (z * torch.exp(log_scale) + shift)


class ConditionalRealNVP(nn.Module if nn is not None else object):
    """Conditional RealNVP flow over the multivariate target vector."""

    def __init__(self, dim: int, cond_dim: int, n_blocks: int, hidden_size: int, n_hidden: int) -> None:
        require_torch()
        super().__init__()
        self.dim = int(dim)
        self.cond_dim = int(cond_dim)
        layers = []
        for block in range(int(n_blocks)):
            mask_values = [float((i + block) % 2) for i in range(self.dim)]
            mask = torch.tensor(mask_values, dtype=torch.float32)
            layers.append(ConditionalAffineCoupling(self.dim, self.cond_dim, hidden_size, n_hidden, mask))
        self.layers = nn.ModuleList(layers)

    def log_prob(self, x: Any, cond: Any) -> Any:
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ldj = layer(z, cond)
            log_det = log_det + ldj
        base = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return base.log_prob(z).sum(dim=-1) + log_det

    def sample(self, cond: Any) -> Any:
        z = torch.randn(cond.shape[0], self.dim, device=cond.device, dtype=cond.dtype)
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, cond)
        return x


class TempFlowNetwork(nn.Module if nn is not None else object):
    """Recurrent temporal conditioner plus conditional flow output."""

    def __init__(
        self,
        target_dim: int,
        time_feature_dim: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        dropout_rate: float,
        lags_seq: Sequence[int],
        scaling: bool,
        flow_type: str,
        n_blocks: int,
        hidden_size: int,
        n_hidden: int,
        conditioning_length: int,
        min_scale: float,
        target_dim_embedding_dim: int = 1,
    ) -> None:
        require_torch()
        super().__init__()
        if str(flow_type) != "RealNVP":
            raise NotImplementedError("FinProbTS TempFlow currently supports the upstream default flow_type='RealNVP'.")

        self.target_dim = int(target_dim)
        self.time_feature_dim = int(time_feature_dim)
        self.num_layers = int(num_layers)
        self.num_cells = int(num_cells)
        self.cell_type = str(cell_type).upper()
        self.lags_seq = [int(lag) for lag in lags_seq]
        self.max_lag = max(self.lags_seq)
        self.scaling = bool(scaling)
        self.conditioning_length = int(conditioning_length)
        self.min_scale = float(min_scale)
        self.target_dim_embedding_dim = int(target_dim_embedding_dim)

        self.embed = nn.Embedding(self.target_dim, self.target_dim_embedding_dim)
        input_size = (
            self.target_dim * len(self.lags_seq)
            + self.target_dim * self.target_dim_embedding_dim
            + self.time_feature_dim
        )
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}.get(self.cell_type)
        if rnn_cls is None:
            raise ValueError("cell_type must be 'LSTM' or 'GRU'.")
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=self.num_cells,
            num_layers=self.num_layers,
            dropout=float(dropout_rate) if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj_dist_args = nn.Linear(self.num_cells, self.conditioning_length)
        self.flow = ConditionalRealNVP(
            dim=self.target_dim,
            cond_dim=self.conditioning_length,
            n_blocks=n_blocks,
            hidden_size=hidden_size,
            n_hidden=n_hidden,
        )

    def _initial_state(self, batch_size: int, device: Any) -> Any:
        shape = (self.num_layers, int(batch_size), self.num_cells)
        h = torch.zeros(shape, device=device)
        if self.cell_type == "LSTM":
            return h, torch.zeros(shape, device=device)
        return h

    @staticmethod
    def _repeat_state(state: Any, repeats: int) -> Any:
        if isinstance(state, tuple):
            return tuple(component.repeat_interleave(repeats, dim=1) for component in state)
        return state.repeat_interleave(repeats, dim=1)

    def _compute_scale(self, past_target: Any, past_observed: Any) -> Any:
        if not self.scaling:
            return torch.ones(
                past_target.shape[0],
                1,
                past_target.shape[-1],
                device=past_target.device,
                dtype=past_target.dtype,
            )
        observed_count = past_observed.sum(dim=1)
        denom = observed_count.clamp_min(1.0)
        mean_abs = (past_target.abs() * past_observed).sum(dim=1) / denom
        scale = torch.where(
            observed_count > 0,
            mean_abs.clamp_min(self.min_scale),
            torch.ones_like(mean_abs),
        )
        return scale.unsqueeze(1)

    def _target_dimension_embeddings(self, batch_size: int, device: Any) -> Any:
        index = torch.arange(self.target_dim, device=device)
        embedded = self.embed(index).reshape(1, -1)
        return embedded.expand(batch_size, -1)

    def _features_from_indices(
        self,
        history: Any,
        step_index: int,
        scale: Any,
        time_feat: Any,
        target_embeddings: Any,
    ) -> Any:
        lagged = [history[:, step_index - lag, :] / scale.squeeze(1) for lag in self.lags_seq]
        return torch.cat((torch.cat(lagged, dim=-1), target_embeddings, time_feat), dim=-1)

    def _features_from_tail(self, history: Any, scale: Any, time_feat: Any, target_embeddings: Any) -> Any:
        lagged = [history[:, -lag, :] / scale.squeeze(1) for lag in self.lags_seq]
        return torch.cat((torch.cat(lagged, dim=-1), target_embeddings, time_feat), dim=-1)

    def _condition_from_rnn(self, rnn_output: Any) -> Any:
        return self.proj_dist_args(rnn_output)

    def _encode_context(self, past_target: Any, past_time_feat: Any, scale: Any) -> Any:
        if past_target.shape[1] < self.max_lag:
            raise ValueError(
                f"context_length={past_target.shape[1]} must be at least max(lags_seq)={self.max_lag}."
            )
        state = self._initial_state(past_target.shape[0], past_target.device)
        if past_target.shape[1] == self.max_lag:
            return state
        target_embeddings = self._target_dimension_embeddings(past_target.shape[0], past_target.device)
        inputs = [
            self._features_from_indices(
                past_target,
                step_index=t,
                scale=scale,
                time_feat=past_time_feat[:, t, :],
                target_embeddings=target_embeddings,
            )
            for t in range(self.max_lag, past_target.shape[1])
        ]
        _, state = self.rnn(torch.stack(inputs, dim=1), state)
        return state

    def loss(self, batch: Dict[str, Any]) -> Any:
        past = batch["past_target"]
        future = batch["future_target"]
        past_observed = batch["past_observed_values"]
        future_observed = batch["future_observed_values"]
        scale = self._compute_scale(past, past_observed)
        scale_log_det = torch.log(scale.squeeze(1).clamp_min(self.min_scale)).sum(dim=-1)
        target_embeddings = self._target_dimension_embeddings(past.shape[0], past.device)
        state = self._initial_state(past.shape[0], past.device)

        weighted_losses = []
        weights = []
        history = past
        for step_index in range(self.max_lag, past.shape[1]):
            step_input = self._features_from_indices(
                history,
                step_index=step_index,
                scale=scale,
                time_feat=batch["past_time_feat"][:, step_index, :],
                target_embeddings=target_embeddings,
            ).unsqueeze(1)
            rnn_output, state = self.rnn(step_input, state)
            cond = self._condition_from_rnn(rnn_output[:, 0, :])
            target_scaled = past[:, step_index, :] / scale.squeeze(1)
            nll = -self.flow.log_prob(target_scaled, cond) + scale_log_det
            observed_weight = past_observed[:, step_index, :].min(dim=-1).values
            weighted_losses.append(nll * observed_weight)
            weights.append(observed_weight)

        for step in range(future.shape[1]):
            step_input = self._features_from_tail(
                history,
                scale=scale,
                time_feat=batch["future_time_feat"][:, step, :],
                target_embeddings=target_embeddings,
            ).unsqueeze(1)
            rnn_output, state = self.rnn(step_input, state)
            cond = self._condition_from_rnn(rnn_output[:, 0, :])
            target_scaled = future[:, step, :] / scale.squeeze(1)
            nll = -self.flow.log_prob(target_scaled, cond) + scale_log_det
            observed_weight = future_observed[:, step, :].min(dim=-1).values
            weighted_losses.append(nll * observed_weight)
            weights.append(observed_weight)
            history = torch.cat((history, future[:, step : step + 1, :]), dim=1)

        loss_sum = torch.stack(weighted_losses, dim=0).sum()
        weight_sum = torch.stack(weights, dim=0).sum()
        if float(weight_sum.detach().cpu()) <= 0.0:
            return loss_sum * 0.0
        return loss_sum / weight_sum

    def sample(self, batch: Dict[str, Any], num_samples: int) -> Any:
        past = batch["past_target"]
        scale = self._compute_scale(past, batch["past_observed_values"])
        state = self._encode_context(past, batch["past_time_feat"], scale)

        repeats = int(num_samples)
        history = past.repeat_interleave(repeats, dim=0)
        repeated_scale = scale.repeat_interleave(repeats, dim=0)
        repeated_state = self._repeat_state(state, repeats)
        repeated_future_time_feat = batch["future_time_feat"].repeat_interleave(repeats, dim=0)
        target_embeddings = self._target_dimension_embeddings(history.shape[0], history.device)

        sample_steps = []
        for step in range(repeated_future_time_feat.shape[1]):
            step_input = self._features_from_tail(
                history,
                scale=repeated_scale,
                time_feat=repeated_future_time_feat[:, step, :],
                target_embeddings=target_embeddings,
            ).unsqueeze(1)
            rnn_output, repeated_state = self.rnn(step_input, repeated_state)
            cond = self._condition_from_rnn(rnn_output[:, 0, :])
            scaled_sample = self.flow.sample(cond)
            sample = scaled_sample * repeated_scale.squeeze(1)
            sample_steps.append(sample)
            history = torch.cat((history, sample.unsqueeze(1)), dim=1)

        flat_samples = torch.stack(sample_steps, dim=1)
        return flat_samples.reshape(past.shape[0], repeats, -1, self.target_dim)


class TempFlowForecastModel(BaseProbForecastModel):
    """TempFlow model with FinProbTS fit/predict/save/load API."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        freq: Optional[str] = None,
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: Sequence[int] = (1,),
        embedding_dimension: int = 5,
        flow_type: str = "RealNVP",
        n_blocks: int = 3,
        hidden_size: int = 100,
        n_hidden: int = 2,
        conditioning_length: int = 200,
        dequantize: bool = False,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[Sequence[int]] = None,
        time_features: Optional[Any] = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        lr: float = 1e-3,
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.0,
        gradient_clip_val: Optional[float] = 10.0,
        patience: Optional[int] = 10,
        device: str = "auto",
        seed: Optional[int] = None,
        min_scale: float = 1e-6,
        verbose: bool = False,
        rnn_type: Optional[str] = None,
        dropout: Optional[float] = None,
        flow_hidden_size: Optional[int] = None,
        hidden_size_flow: Optional[int] = None,
        flow_n_hidden: Optional[int] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if rnn_type is not None:
            cell_type = str(rnn_type)
        if dropout is not None:
            dropout_rate = float(dropout)
        if flow_hidden_size is not None:
            hidden_size = int(flow_hidden_size)
        if hidden_size_flow is not None:
            hidden_size = int(hidden_size_flow)
        if flow_n_hidden is not None:
            n_hidden = int(flow_n_hidden)
        if learning_rate is not None:
            lr = float(learning_rate)
        if optim_kwargs:
            lr = float(optim_kwargs.get("lr", lr))
            weight_decay = float(optim_kwargs.get("weight_decay", weight_decay))
            patience = optim_kwargs.get("patience", patience)
        if time_features is not None:
            raise NotImplementedError("Custom TempFlow time_features are not wired into FinProbTS yet.")
        if bool(dequantize):
            raise NotImplementedError("TempFlow dequantize=True is not implemented for continuous FinProbTS panels.")
        if bool(pick_incomplete):
            raise NotImplementedError("TempFlow pick_incomplete=True is an upstream sampler option; FinProbTS uses complete rolling windows.")
        if str(flow_type) != "RealNVP":
            raise NotImplementedError("FinProbTS TempFlow currently supports the upstream default flow_type='RealNVP' only.")

        self.input_size = None if input_size is None else int(input_size)
        self.freq = None if freq is None else str(freq)
        self.prediction_length = None if prediction_length is None else int(prediction_length)
        self.target_dim = None if target_dim is None else int(target_dim)
        self.context_length = None if context_length is None else int(context_length)
        self.num_layers = int(num_layers)
        self.num_cells = int(num_cells)
        self.cell_type = str(cell_type).upper()
        self.num_parallel_samples = int(num_parallel_samples)
        self.dropout_rate = float(dropout_rate)
        self.cardinality = [int(item) for item in cardinality]
        self.embedding_dimension = int(embedding_dimension)
        self.flow_type = str(flow_type)
        self.n_blocks = int(n_blocks)
        self.hidden_size = int(hidden_size)
        self.n_hidden = int(n_hidden)
        self.conditioning_length = int(conditioning_length)
        self.dequantize = bool(dequantize)
        self.scaling = bool(scaling)
        self.pick_incomplete = bool(pick_incomplete)
        self.lags_seq = _normalize_lags(lags_seq, self.freq)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.learning_rate = self.lr
        self.weight_decay = float(weight_decay)
        self.gradient_clip_val = None if gradient_clip_val is None else float(gradient_clip_val)
        self.patience = None if patience is None else int(patience)
        self.device_name = device
        self.seed = seed
        self.min_scale = float(min_scale)
        self.verbose = bool(verbose)

        self._network: Optional[TempFlowNetwork] = None
        self._device = None
        self._num_assets: Optional[int] = None
        self._asset_ids: Optional[list[str]] = None
        self._time_feature_dim: Optional[int] = None
        self._effective_input_size: Optional[int] = None
        self._is_fitted = False
        self.training_history: list[Dict[str, float]] = []

    def _init_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "freq": self.freq,
            "prediction_length": self.prediction_length,
            "target_dim": self.target_dim,
            "context_length": self.context_length,
            "num_layers": self.num_layers,
            "num_cells": self.num_cells,
            "cell_type": self.cell_type,
            "num_parallel_samples": self.num_parallel_samples,
            "dropout_rate": self.dropout_rate,
            "cardinality": list(self.cardinality),
            "embedding_dimension": self.embedding_dimension,
            "flow_type": self.flow_type,
            "n_blocks": self.n_blocks,
            "hidden_size": self.hidden_size,
            "n_hidden": self.n_hidden,
            "conditioning_length": self.conditioning_length,
            "dequantize": self.dequantize,
            "scaling": self.scaling,
            "pick_incomplete": self.pick_incomplete,
            "lags_seq": list(self.lags_seq),
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "gradient_clip_val": self.gradient_clip_val,
            "patience": self.patience,
            "device": self.device_name,
            "seed": self.seed,
            "min_scale": self.min_scale,
            "verbose": self.verbose,
        }

    def _validate_data(self, data: RollingWindowDataset) -> None:
        if self.context_length is not None and data.context_length != self.context_length:
            raise ValueError(
                "TempFlow context_length is controlled by the FinProbTS task window. "
                f"Got model context_length={self.context_length} but data has context_length={data.context_length}."
            )
        if self.prediction_length is not None and data.prediction_length != self.prediction_length:
            raise ValueError(
                "TempFlow prediction_length is controlled by the FinProbTS task window. "
                f"Got model prediction_length={self.prediction_length} but data has prediction_length={data.prediction_length}."
            )
        if self.target_dim is not None and data.num_assets != self.target_dim:
            raise ValueError(f"TempFlow target_dim={self.target_dim} does not match data num_assets={data.num_assets}.")
        if data.context_length < max(self.lags_seq):
            raise ValueError(
                f"TempFlow requires context_length >= max(lags_seq). "
                f"Got context_length={data.context_length}, lags_seq={self.lags_seq}."
            )

    def _build_network(self, num_assets: int, time_feature_dim: int) -> None:
        self._num_assets = int(num_assets)
        self._time_feature_dim = int(time_feature_dim)
        self._effective_input_size = (
            int(num_assets) * len(self.lags_seq)
            + int(num_assets) * self.embedding_dimension
            + int(time_feature_dim)
        )
        self._device = resolve_torch_device(self.device_name)
        self._network = TempFlowNetwork(
            target_dim=num_assets,
            time_feature_dim=time_feature_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            flow_type=self.flow_type,
            n_blocks=self.n_blocks,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            conditioning_length=self.conditioning_length,
            min_scale=self.min_scale,
            target_dim_embedding_dim=self.embedding_dimension,
        ).to(self._device)

    def _make_loader(self, data: RollingWindowDataset, shuffle: bool) -> Any:
        return make_torch_data_loader(data, self.batch_size, shuffle, scaler=None, include_time_features=True)

    def _evaluate(self, loader: Any) -> float:
        assert self._network is not None and self._device is not None
        self._network.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                loss = self._network.loss(batch)
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
        sample_loader = self._make_loader(train_data, shuffle=False)
        first_batch = next(iter_torch_batches(sample_loader, resolve_torch_device("cpu")))
        self._build_network(train_data.num_assets, int(first_batch["past_time_feat"].shape[-1]))

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
            for batch in iter_torch_batches(train_loader, self._device):
                optimizer.zero_grad()
                loss = self._network.loss(batch)
                loss.backward()
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.gradient_clip_val)
                optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
            train_loss = total / max(count, 1)
            val_loss = self._evaluate(val_loader) if val_loader is not None else train_loss
            self.training_history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            if self.verbose:
                print(f"TempFlow epoch {epoch}/{self.max_epochs}: train={train_loss:.6f} val={val_loss:.6f}")
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
            for batch in iter_torch_batches(loader, self._device):
                chunks.append(self._network.sample(batch, int(num_samples)).cpu().numpy())
        samples = np.concatenate(chunks, axis=0)
        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "tempflow",
                "implementation": "finprobts_native_tempflow",
                "reference": "PyTorchTS TempFlowEstimator",
                "seed": self.seed,
                "flow_type": self.flow_type,
                "n_blocks": self.n_blocks,
                "hidden_size": self.hidden_size,
                "num_cells": self.num_cells,
                "num_layers": self.num_layers,
                "lags_seq": list(self.lags_seq),
                "conditioning_length": self.conditioning_length,
                "model_internal_scaling": self.scaling,
                "effective_input_size": self._effective_input_size,
                "num_parallel_samples_default": self.num_parallel_samples,
                "training_history": list(self.training_history),
            },
        )

    def save(self, path: str) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("Cannot save an unfitted TempFlowForecastModel.")
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "init_params": self._init_params(),
                "model_state": self._network.state_dict(),
                "num_assets": self._num_assets,
                "asset_ids": self._asset_ids,
                "time_feature_dim": self._time_feature_dim,
                "effective_input_size": self._effective_input_size,
                "training_history": self.training_history,
                "is_fitted": self._is_fitted,
            },
            output_dir / "model.pt",
        )

    @classmethod
    def load(cls, path: str) -> "TempFlowForecastModel":
        require_torch()
        try:
            payload = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(Path(path) / "model.pt", map_location="cpu")
        model = cls(**payload["init_params"])
        model._asset_ids = [str(item) for item in payload.get("asset_ids", [])]
        model._build_network(int(payload["num_assets"]), int(payload.get("time_feature_dim", 4)))
        model._network.load_state_dict(payload["model_state"])
        model._effective_input_size = payload.get("effective_input_size")
        model.training_history = list(payload.get("training_history", []))
        model._is_fitted = bool(payload.get("is_fitted", True))
        return model
