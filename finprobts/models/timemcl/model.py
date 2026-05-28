"""TimeMCL forecaster for the FinProbTS rolling-window contract.

This module implements a native PyTorch TimeMCL-style model following the
public TimeMCL design: lagged multivariate autoregressive inputs, recurrent
temporal conditioning, target-dimension embeddings, mean scaling, multiple
hypothesis heads, score heads, and WTA/relaxed-WTA/annealed-WTA losses.
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


class MeanLayer(nn.Module if nn is not None else object):
    """Mean aggregation layer for TimeMCL score heads."""

    def __init__(self, dim: int, keepdim: bool = False) -> None:
        require_torch()
        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)

    def forward(self, x: Any) -> Any:
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class TimeMCLHeads(nn.Module if nn is not None else object):
    """Multiple prediction heads plus score heads trained with MCL losses."""

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_hypotheses: int,
        mcl_loss_type: str,
        score_loss_weight: float,
        wta_mode: str,
        wta_mode_params: Optional[Dict[str, Any]],
        single_linear_layer: bool,
    ) -> None:
        require_torch()
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hypotheses = int(num_hypotheses)
        self.mcl_loss_type = str(mcl_loss_type)
        self.score_loss_weight = float(score_loss_weight)
        self.wta_mode = str(wta_mode)
        self.wta_mode_params = dict(wta_mode_params or {})
        if self.num_hypotheses <= 0:
            raise ValueError("num_hypotheses must be positive.")

        if bool(single_linear_layer):
            self.prediction_heads = nn.ModuleList(
                [nn.Linear(self.cond_dim, self.target_dim) for _ in range(self.num_hypotheses)]
            )
        else:
            self.prediction_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.cond_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.1),
                        nn.Linear(self.hidden_dim, self.target_dim),
                    )
                    for _ in range(self.num_hypotheses)
                ]
            )
        self.score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.cond_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(self.num_hypotheses)
            ]
        )
        self.score_aggregator = MeanLayer(dim=1)

    def update_wta_mode_params(self, params: Dict[str, Any], mode: Optional[str] = None) -> None:
        self.wta_mode_params.update(params)
        if mode is not None:
            self.wta_mode = str(mode)

    def forward(self, cond: Any) -> tuple[Any, Any]:
        batch_size, sequence_length, cond_dim = cond.shape
        flat_cond = cond.reshape(batch_size * sequence_length, cond_dim)

        predictions = []
        for head in self.prediction_heads:
            predictions.append(head(flat_cond).reshape(batch_size, sequence_length, self.target_dim))
        prediction_list = torch.stack(predictions, dim=1)

        scores = []
        for head in self.score_heads:
            score = head(flat_cond).reshape(batch_size, sequence_length)
            scores.append(self.score_aggregator(score))
        score_list = torch.stack(scores, dim=1).clamp_min(1e-8)
        return prediction_list, score_list

    def _pairwise_distance(self, predictions: Any, target: Any, observed: Any) -> Any:
        error = (predictions - target.unsqueeze(1)).square()
        observed_assets = observed.unsqueeze(1)
        denom = observed.sum(dim=-1).clamp_min(1.0)
        per_step = (error * observed_assets).sum(dim=-1) / denom.unsqueeze(1)
        time_weight = observed.min(dim=-1).values
        denom_time = time_weight.sum(dim=-1).clamp_min(1.0)
        return (per_step * time_weight.unsqueeze(1)).sum(dim=-1) / denom_time.unsqueeze(1)

    def _wta_loss(self, pairwise_distance: Any) -> tuple[Any, Any]:
        mode = self.wta_mode
        if mode == "wta":
            return pairwise_distance.min(dim=1)
        if mode == "relaxed-wta":
            if self.num_hypotheses <= 1:
                raise ValueError("relaxed-wta requires at least two hypotheses.")
            epsilon = float(self.wta_mode_params.get("epsilon", 0.1))
            winner, assignment = pairwise_distance.min(dim=1)
            loss = (1.0 - epsilon * self.num_hypotheses / (self.num_hypotheses - 1.0)) * winner
            loss = loss + (epsilon / (self.num_hypotheses - 1.0)) * pairwise_distance.sum(dim=1)
            return loss, assignment
        if mode == "awta":
            temperature = max(float(self.wta_mode_params.get("temperature", 1.0)), 1e-8)
            assignment = pairwise_distance.min(dim=1).indices
            weights = torch.softmax(-pairwise_distance / temperature, dim=1).detach()
            return (weights * pairwise_distance).mean(dim=1), assignment
        raise ValueError("wta_mode must be one of 'wta', 'relaxed-wta', or 'awta'.")

    def loss(self, cond: Any, target: Any, observed: Any) -> tuple[Any, Any, Any]:
        predictions, scores = self.forward(cond)
        if self.mcl_loss_type == "min_ext_sum":
            pairwise_distance = self._pairwise_distance(predictions, target, observed)
            mcl_loss, assignment = self._wta_loss(pairwise_distance)
        elif self.mcl_loss_type == "min_in_sum":
            error = (predictions - target.unsqueeze(1)).square()
            denom = observed.sum(dim=-1).clamp_min(1.0)
            per_step = (error * observed.unsqueeze(1)).sum(dim=-1) / denom.unsqueeze(1)
            min_step, assignment_step = per_step.min(dim=1)
            time_weight = observed.min(dim=-1).values
            mcl_loss = (min_step * time_weight).sum(dim=-1) / time_weight.sum(dim=-1).clamp_min(1.0)
            assignment = torch.mode(assignment_step, dim=1).values
        else:
            raise ValueError("mcl_loss_type must be 'min_ext_sum' or 'min_in_sum'.")

        target_assignment = F.one_hot(assignment, num_classes=self.num_hypotheses).float()
        score_loss = F.binary_cross_entropy(scores.clamp(1e-8, 1.0 - 1e-8), target_assignment)
        total = mcl_loss.mean() + self.score_loss_weight * score_loss
        return total, assignment, score_loss

    def sample(self, cond: Any) -> tuple[Any, Any]:
        predictions, scores = self.forward(cond)
        return predictions, scores


class TimeMCLNetwork(nn.Module if nn is not None else object):
    """RNN temporal conditioner plus TimeMCL multi-hypothesis output."""

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
        scaler_type: str,
        min_scale: float,
        embedding_dimension: int,
        conditioning_length: int,
        num_hypotheses: int,
        mcl_hidden_dim: int,
        mcl_loss_type: str,
        score_loss_weight: float,
        wta_mode: str,
        wta_mode_params: Optional[Dict[str, Any]],
        single_linear_layer: bool,
    ) -> None:
        require_torch()
        super().__init__()
        self.target_dim = int(target_dim)
        self.time_feature_dim = int(time_feature_dim)
        self.num_layers = int(num_layers)
        self.num_cells = int(num_cells)
        self.cell_type = str(cell_type).upper()
        self.lags_seq = [int(lag) for lag in lags_seq]
        self.max_lag = max(self.lags_seq)
        self.scaling = bool(scaling)
        self.scaler_type = str(scaler_type)
        self.min_scale = float(min_scale)
        self.embed_dim = int(embedding_dimension)
        self.num_hypotheses = int(num_hypotheses)

        if self.scaler_type not in {"mean", "nops"}:
            raise NotImplementedError("FinProbTS TimeMCL currently supports scaler_type='mean' or 'nops'.")

        self.embed = nn.Embedding(self.target_dim, self.embed_dim) if self.embed_dim > 0 else None
        input_size = self.target_dim * len(self.lags_seq) + self.target_dim * self.embed_dim + self.time_feature_dim
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
        self.proj_dist_args = nn.Linear(self.num_cells, int(conditioning_length))
        self.mcl = TimeMCLHeads(
            cond_dim=int(conditioning_length),
            target_dim=self.target_dim,
            hidden_dim=int(mcl_hidden_dim),
            num_hypotheses=self.num_hypotheses,
            mcl_loss_type=mcl_loss_type,
            score_loss_weight=score_loss_weight,
            wta_mode=wta_mode,
            wta_mode_params=wta_mode_params,
            single_linear_layer=single_linear_layer,
        )

    def update_wta_mode_params(self, params: Dict[str, Any], mode: Optional[str] = None) -> None:
        self.mcl.update_wta_mode_params(params, mode=mode)

    def _initial_state(self, batch_size: int, device: Any) -> Any:
        shape = (self.num_layers, int(batch_size), self.num_cells)
        h = torch.zeros(shape, device=device)
        if self.cell_type == "LSTM":
            return h, torch.zeros(shape, device=device)
        return h

    @staticmethod
    def _clone_state(state: Any) -> Any:
        if isinstance(state, tuple):
            return tuple(component.clone() for component in state)
        return state.clone()

    def _compute_scale(self, past_target: Any, past_observed: Any) -> Any:
        if (not self.scaling) or self.scaler_type == "nops":
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

    def _target_dimension_embeddings(self, batch_size: int, device: Any) -> list[Any]:
        if self.embed is None:
            return []
        index = torch.arange(self.target_dim, device=device)
        embedded = self.embed(index).reshape(1, -1)
        return [embedded.expand(batch_size, -1)]

    def _features_from_indices(
        self,
        history: Any,
        step_index: int,
        scale: Any,
        time_feat: Any,
        target_embeddings: Sequence[Any],
    ) -> Any:
        lagged = [history[:, step_index - lag, :] / scale.squeeze(1) for lag in self.lags_seq]
        return torch.cat((torch.cat(lagged, dim=-1), *target_embeddings, time_feat), dim=-1)

    def _features_from_tail(self, history: Any, scale: Any, time_feat: Any, target_embeddings: Sequence[Any]) -> Any:
        lagged = [history[:, -lag, :] / scale.squeeze(1) for lag in self.lags_seq]
        return torch.cat((torch.cat(lagged, dim=-1), *target_embeddings, time_feat), dim=-1)

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

    def loss(self, batch: Dict[str, Any]) -> tuple[Any, Any, Any]:
        past = batch["past_target"]
        future = batch["future_target"]
        past_observed = batch["past_observed_values"]
        future_observed = batch["future_observed_values"]
        scale = self._compute_scale(past, past_observed)
        target_embeddings = self._target_dimension_embeddings(past.shape[0], past.device)
        state = self._initial_state(past.shape[0], past.device)

        conditions = []
        targets = []
        observed = []
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
            conditions.append(self._condition_from_rnn(rnn_output[:, 0, :]))
            targets.append(past[:, step_index, :])
            observed.append(past_observed[:, step_index, :])

        for step in range(future.shape[1]):
            step_input = self._features_from_tail(
                history,
                scale=scale,
                time_feat=batch["future_time_feat"][:, step, :],
                target_embeddings=target_embeddings,
            ).unsqueeze(1)
            rnn_output, state = self.rnn(step_input, state)
            conditions.append(self._condition_from_rnn(rnn_output[:, 0, :]))
            targets.append(future[:, step, :])
            observed.append(future_observed[:, step, :])
            history = torch.cat((history, future[:, step : step + 1, :]), dim=1)

        cond_seq = torch.stack(conditions, dim=1)
        target_seq = torch.stack(targets, dim=1) / scale
        observed_seq = torch.stack(observed, dim=1)
        return self.mcl.loss(cond_seq, target_seq, observed_seq)

    def sample_hypotheses(self, batch: Dict[str, Any]) -> tuple[Any, Any]:
        past = batch["past_target"]
        scale = self._compute_scale(past, batch["past_observed_values"])
        initial_state = self._encode_context(past, batch["past_time_feat"], scale)
        target_embeddings = self._target_dimension_embeddings(past.shape[0], past.device)

        all_hypotheses = []
        all_scores = []
        for hyp_idx in range(self.num_hypotheses):
            history = past
            state = self._clone_state(initial_state)
            hyp_steps = []
            score_steps = []
            for step in range(batch["future_time_feat"].shape[1]):
                step_input = self._features_from_tail(
                    history,
                    scale=scale,
                    time_feat=batch["future_time_feat"][:, step, :],
                    target_embeddings=target_embeddings,
                ).unsqueeze(1)
                rnn_output, state = self.rnn(step_input, state)
                cond = self._condition_from_rnn(rnn_output)
                predictions, scores = self.mcl.sample(cond)
                scaled_sample = predictions[:, hyp_idx, 0, :]
                sample = scaled_sample * scale.squeeze(1)
                hyp_steps.append(sample)
                score_steps.append(scores[:, hyp_idx])
                history = torch.cat((history, sample.unsqueeze(1)), dim=1)
            all_hypotheses.append(torch.stack(hyp_steps, dim=1))
            all_scores.append(torch.stack(score_steps, dim=1).mean(dim=1))

        hypotheses = torch.stack(all_hypotheses, dim=1)
        scores = torch.stack(all_scores, dim=1).clamp_min(1e-8)
        return hypotheses, scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)


class TimeMCLForecastModel(BaseProbForecastModel):
    """TimeMCL model with FinProbTS fit/predict/save/load API."""

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
        embedding_dimension: int = 0,
        conditioning_length: int = 100,
        num_hypotheses: int = 4,
        mcl_hidden_dim: int = 300,
        mcl_loss_type: str = "min_ext_sum",
        score_loss_weight: float = 0.5,
        wta_mode: str = "wta",
        wta_mode_params: Optional[Dict[str, Any]] = None,
        single_linear_layer: bool = True,
        backbone_deleted: bool = True,
        scaling: bool = True,
        scaler_type: str = "mean",
        pick_incomplete: bool = False,
        lags_seq: Optional[Sequence[int]] = None,
        time_features: Optional[Any] = None,
        sample_hyps: bool = True,
        sample_noise_std: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 100,
        lr: float = 1e-3,
        learning_rate: Optional[float] = None,
        weight_decay: float = 1e-8,
        gradient_clip_val: Optional[float] = 10.0,
        patience: Optional[int] = 10,
        device: str = "auto",
        seed: Optional[int] = None,
        min_scale: float = 1e-6,
        verbose: bool = False,
        hidden_size: Optional[int] = None,
        rnn_type: Optional[str] = None,
        dropout: Optional[float] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if hidden_size is not None:
            num_cells = int(hidden_size)
        if rnn_type is not None:
            cell_type = str(rnn_type)
        if dropout is not None:
            dropout_rate = float(dropout)
        if learning_rate is not None:
            lr = float(learning_rate)
        if optim_kwargs:
            lr = float(optim_kwargs.get("lr", lr))
            weight_decay = float(optim_kwargs.get("weight_decay", weight_decay))
            patience = optim_kwargs.get("patience", patience)
        if time_features is not None:
            raise NotImplementedError("Custom TimeMCL time_features are not wired into FinProbTS yet.")
        if bool(pick_incomplete):
            raise NotImplementedError("TimeMCL pick_incomplete=True is an upstream sampler option; FinProbTS uses complete rolling windows.")
        if not bool(backbone_deleted):
            raise NotImplementedError("FinProbTS TimeMCL currently implements the upstream default backbone_deleted=True.")

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
        self.embedding_dimension = int(embedding_dimension)
        self.conditioning_length = int(conditioning_length)
        self.num_hypotheses = int(num_hypotheses)
        self.mcl_hidden_dim = int(mcl_hidden_dim)
        self.mcl_loss_type = str(mcl_loss_type)
        self.score_loss_weight = float(score_loss_weight)
        self.wta_mode = str(wta_mode)
        self.wta_mode_params = dict(wta_mode_params or {})
        self.single_linear_layer = bool(single_linear_layer)
        self.backbone_deleted = bool(backbone_deleted)
        self.scaling = bool(scaling)
        self.scaler_type = str(scaler_type)
        self.pick_incomplete = bool(pick_incomplete)
        self.lags_seq = _normalize_lags(lags_seq, self.freq)
        self.sample_hyps = bool(sample_hyps)
        self.sample_noise_std = float(sample_noise_std)
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

        self._network: Optional[TimeMCLNetwork] = None
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
            "embedding_dimension": self.embedding_dimension,
            "conditioning_length": self.conditioning_length,
            "num_hypotheses": self.num_hypotheses,
            "mcl_hidden_dim": self.mcl_hidden_dim,
            "mcl_loss_type": self.mcl_loss_type,
            "score_loss_weight": self.score_loss_weight,
            "wta_mode": self.wta_mode,
            "wta_mode_params": dict(self.wta_mode_params),
            "single_linear_layer": self.single_linear_layer,
            "backbone_deleted": self.backbone_deleted,
            "scaling": self.scaling,
            "scaler_type": self.scaler_type,
            "pick_incomplete": self.pick_incomplete,
            "lags_seq": list(self.lags_seq),
            "sample_hyps": self.sample_hyps,
            "sample_noise_std": self.sample_noise_std,
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
                "TimeMCL context_length is controlled by the FinProbTS task window. "
                f"Got model context_length={self.context_length} but data has context_length={data.context_length}."
            )
        if self.prediction_length is not None and data.prediction_length != self.prediction_length:
            raise ValueError(
                "TimeMCL prediction_length is controlled by the FinProbTS task window. "
                f"Got model prediction_length={self.prediction_length} but data has prediction_length={data.prediction_length}."
            )
        if self.target_dim is not None and data.num_assets != self.target_dim:
            raise ValueError(f"TimeMCL target_dim={self.target_dim} does not match data num_assets={data.num_assets}.")
        if data.context_length < max(self.lags_seq):
            raise ValueError(
                f"TimeMCL requires context_length >= max(lags_seq). "
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
        if self.input_size is not None and self.input_size != self._effective_input_size:
            raise ValueError(
                f"Configured input_size={self.input_size} does not match FinProbTS-derived "
                f"TimeMCL input_size={self._effective_input_size}."
            )
        self._device = resolve_torch_device(self.device_name)
        self._network = TimeMCLNetwork(
            target_dim=num_assets,
            time_feature_dim=time_feature_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            scaler_type=self.scaler_type,
            min_scale=self.min_scale,
            embedding_dimension=self.embedding_dimension,
            conditioning_length=self.conditioning_length,
            num_hypotheses=self.num_hypotheses,
            mcl_hidden_dim=self.mcl_hidden_dim,
            mcl_loss_type=self.mcl_loss_type,
            score_loss_weight=self.score_loss_weight,
            wta_mode=self.wta_mode,
            wta_mode_params=self.wta_mode_params,
            single_linear_layer=self.single_linear_layer,
        ).to(self._device)

    def _make_loader(self, data: RollingWindowDataset, shuffle: bool) -> Any:
        return make_torch_data_loader(data, self.batch_size, shuffle, scaler=None, include_time_features=True)

    def _temperature_for_epoch(self, epoch: int) -> Optional[float]:
        if self.wta_mode != "awta":
            return None
        params = self.wta_mode_params
        initial = float(params.get("temperature_ini", params.get("temperature", 10.0)))
        scheduler_mode = str(params.get("scheduler_mode", "exponential"))
        if scheduler_mode == "constant":
            temperature = initial
        elif scheduler_mode == "linear":
            temperature = initial - initial * float(epoch - 1) / max(float(self.max_epochs), 1.0)
        elif scheduler_mode == "exponential":
            temperature = initial * float(params.get("temperature_decay", 0.95)) ** int(epoch - 1)
        else:
            raise ValueError("wta_mode_params.scheduler_mode must be 'constant', 'linear', or 'exponential'.")
        return float(temperature)

    def _update_wta_for_epoch(self, epoch: int) -> None:
        if self._network is None:
            return
        temperature = self._temperature_for_epoch(epoch)
        if temperature is None:
            return
        params = dict(self.wta_mode_params)
        limit = float(params.get("temperature_lim", 5e-4))
        mode = "awta"
        if temperature < limit:
            if bool(params.get("wta_after_temperature_lim", True)):
                mode = "wta"
                temperature = limit
            else:
                temperature = limit
        params["temperature"] = temperature
        self._network.update_wta_mode_params(params, mode=mode)

    def _evaluate(self, loader: Any) -> float:
        assert self._network is not None and self._device is not None
        self._network.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                loss = self._network.loss(batch)[0]
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
            self._update_wta_for_epoch(epoch)
            self._network.train()
            total = 0.0
            count = 0
            for batch in iter_torch_batches(train_loader, self._device):
                optimizer.zero_grad()
                loss, _, _ = self._network.loss(batch)
                loss.backward()
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.gradient_clip_val)
                optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
            train_loss = total / max(count, 1)
            val_loss = self._evaluate(val_loader) if val_loader is not None else train_loss
            history_row = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
            if self.wta_mode == "awta" and self._network is not None:
                history_row["temperature"] = float(self._network.mcl.wta_mode_params.get("temperature", np.nan))
            self.training_history.append(history_row)
            if self.verbose:
                print(f"TimeMCL epoch {epoch}/{self.max_epochs}: train={train_loss:.6f} val={val_loss:.6f}")
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

    @staticmethod
    def _resample_hypotheses(hypotheses: Any, scores: Any, num_samples: int, sample_hyps: bool) -> Any:
        batch_size, num_hypotheses, prediction_length, num_assets = hypotheses.shape
        if sample_hyps:
            indices = torch.multinomial(scores, num_samples=int(num_samples), replacement=True)
        else:
            base = torch.arange(int(num_samples), device=hypotheses.device) % num_hypotheses
            indices = base.unsqueeze(0).expand(batch_size, -1)
        return torch.gather(
            hypotheses,
            dim=1,
            index=indices.reshape(batch_size, int(num_samples), 1, 1).expand(-1, -1, prediction_length, num_assets),
        )

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
        score_chunks = []
        self._network.eval()
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                hypotheses, scores = self._network.sample_hypotheses(batch)
                gathered = self._resample_hypotheses(hypotheses, scores, int(num_samples), self.sample_hyps)
                if self.sample_noise_std > 0.0:
                    gathered = gathered + self.sample_noise_std * torch.randn_like(gathered)
                chunks.append(gathered.cpu().numpy())
                score_chunks.append(scores.cpu().numpy())
        samples = np.concatenate(chunks, axis=0)
        scores_np = np.concatenate(score_chunks, axis=0)
        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "timemcl",
                "implementation": "finprobts_native_timemcl",
                "reference": "Victorletzelter/timeMCL",
                "seed": self.seed,
                "num_hypotheses": self.num_hypotheses,
                "mcl_hidden_dim": self.mcl_hidden_dim,
                "mcl_loss_type": self.mcl_loss_type,
                "score_loss_weight": self.score_loss_weight,
                "wta_mode": self.wta_mode,
                "wta_mode_params": dict(self.wta_mode_params),
                "sample_hyps": self.sample_hyps,
                "mean_hypothesis_scores": scores_np.mean(axis=0).tolist(),
                "num_cells": self.num_cells,
                "num_layers": self.num_layers,
                "conditioning_length": self.conditioning_length,
                "lags_seq": list(self.lags_seq),
                "model_internal_scaling": self.scaling,
                "scaler_type": self.scaler_type,
                "effective_input_size": self._effective_input_size,
                "num_parallel_samples_default": self.num_parallel_samples,
                "training_history": list(self.training_history),
            },
        )

    def save(self, path: str) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("Cannot save an unfitted TimeMCLForecastModel.")
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
    def load(cls, path: str) -> "TimeMCLForecastModel":
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
