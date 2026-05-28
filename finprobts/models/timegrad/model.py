"""TimeGrad forecaster for the FinProbTS rolling-window contract.

This module implements a native PyTorch TimeGrad-style model following the
PyTorchTS design: lagged multivariate autoregressive inputs, recurrent temporal
conditioning, target-dimension embeddings, mean scaling, and a conditional
Gaussian diffusion output with a residual convolutional epsilon network.
"""

from __future__ import annotations

import copy
import math
from functools import partial
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


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = int(timesteps) + 1
    x = np.linspace(0, int(timesteps), steps)
    alphas_cumprod = np.cos(((x / int(timesteps)) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0, 0.999)


def _extract(values: Any, timestep: Any, target_shape: Sequence[int]) -> Any:
    gathered = values.gather(-1, timestep)
    return gathered.reshape(timestep.shape[0], *((1,) * (len(target_shape) - 1)))


class DiffusionEmbedding(nn.Module if nn is not None else object):
    """Sinusoidal diffusion-step embedding with two projections."""

    def __init__(self, dim: int, proj_dim: int, max_steps: int = 500) -> None:
        require_torch()
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(int(dim), int(max_steps)), persistent=False)
        self.projection1 = nn.Linear(int(dim) * 2, int(proj_dim))
        self.projection2 = nn.Linear(int(proj_dim), int(proj_dim))

    @staticmethod
    def _build_embedding(dim: int, max_steps: int) -> Any:
        steps = torch.arange(int(max_steps), dtype=torch.float32).unsqueeze(1)
        dims = torch.arange(int(dim), dtype=torch.float32).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / float(dim))
        return torch.cat((torch.sin(table), torch.cos(table)), dim=1)

    def forward(self, diffusion_step: Any) -> Any:
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))


class TimeGradResidualBlock(nn.Module if nn is not None else object):
    """Dilated residual block used by the TimeGrad epsilon network."""

    def __init__(self, hidden_size: int, residual_channels: int, dilation: int) -> None:
        require_torch()
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            int(residual_channels),
            2 * int(residual_channels),
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(int(hidden_size), int(residual_channels))
        self.conditioner_projection = nn.Conv1d(
            1,
            2 * int(residual_channels),
            kernel_size=1,
            padding=2,
            padding_mode="circular",
        )
        self.output_projection = nn.Conv1d(int(residual_channels), 2 * int(residual_channels), kernel_size=1)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x: Any, conditioner: Any, diffusion_step: Any) -> tuple[Any, Any]:
        projected_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        projected_conditioner = self.conditioner_projection(conditioner)
        y = self.dilated_conv(x + projected_step) + projected_conditioner
        gate, filter_part = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_part)
        y = F.leaky_relu(self.output_projection(y), negative_slope=0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class TimeGradConditionUpsampler(nn.Module if nn is not None else object):
    """Map RNN conditioner vectors to the target dimension."""

    def __init__(self, cond_length: int, target_dim: int) -> None:
        require_torch()
        super().__init__()
        hidden = max(1, int(target_dim) // 2)
        self.linear1 = nn.Linear(int(cond_length), hidden)
        self.linear2 = nn.Linear(hidden, int(target_dim))

    def forward(self, x: Any) -> Any:
        x = F.leaky_relu(self.linear1(x), negative_slope=0.4)
        return F.leaky_relu(self.linear2(x), negative_slope=0.4)


class EpsilonTheta(nn.Module if nn is not None else object):
    """TimeGrad residual convolutional denoiser."""

    def __init__(
        self,
        target_dim: int,
        cond_length: int,
        time_emb_dim: int = 16,
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        residual_hidden: int = 64,
        max_steps: int = 500,
    ) -> None:
        require_torch()
        super().__init__()
        self.input_projection = nn.Conv1d(
            1,
            int(residual_channels),
            kernel_size=1,
            padding=2,
            padding_mode="circular",
        )
        self.diffusion_embedding = DiffusionEmbedding(int(time_emb_dim), int(residual_hidden), max_steps=max_steps)
        self.cond_upsampler = TimeGradConditionUpsampler(cond_length=int(cond_length), target_dim=int(target_dim))
        self.residual_layers = nn.ModuleList(
            [
                TimeGradResidualBlock(
                    residual_channels=int(residual_channels),
                    dilation=2 ** (idx % int(dilation_cycle_length)),
                    hidden_size=int(residual_hidden),
                )
                for idx in range(int(residual_layers))
            ]
        )
        self.skip_projection = nn.Conv1d(int(residual_channels), int(residual_channels), kernel_size=3)
        self.output_projection = nn.Conv1d(int(residual_channels), 1, kernel_size=3)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs: Any, time: Any, cond: Any) -> Any:
        x = F.leaky_relu(self.input_projection(inputs), negative_slope=0.4)
        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip_connections = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_up, diffusion_step)
            skip_connections.append(skip)
        x = torch.stack(skip_connections, dim=0).sum(dim=0) / math.sqrt(len(skip_connections))
        x = F.leaky_relu(self.skip_projection(x), negative_slope=0.4)
        return self.output_projection(x)


class GaussianDiffusion(nn.Module if nn is not None else object):
    """Conditional Gaussian diffusion used by TimeGrad."""

    def __init__(
        self,
        denoise_fn: Any,
        input_size: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        diff_steps: int = 100,
        loss_type: str = "l2",
        beta_schedule: str = "linear",
    ) -> None:
        require_torch()
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = int(input_size)
        self.loss_type = str(loss_type).lower()
        betas = self._make_betas(beta_start, beta_end, diff_steps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1.0)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    @staticmethod
    def _make_betas(beta_start: float, beta_end: float, diff_steps: int, beta_schedule: str) -> np.ndarray:
        schedule = str(beta_schedule).lower()
        steps = int(diff_steps)
        if schedule == "linear":
            betas = np.linspace(float(beta_start), float(beta_end), steps)
        elif schedule == "quad":
            betas = np.linspace(float(beta_start) ** 0.5, float(beta_end) ** 0.5, steps) ** 2
        elif schedule == "const":
            betas = float(beta_end) * np.ones(steps)
        elif schedule == "jsd":
            betas = 1.0 / np.linspace(steps, 1, steps)
        elif schedule == "sigmoid":
            grid = np.linspace(-6, 6, steps)
            betas = (float(beta_end) - float(beta_start)) / (np.exp(-grid) + 1.0) + float(beta_start)
        elif schedule == "cosine":
            betas = _cosine_beta_schedule(steps)
        else:
            raise NotImplementedError(f"Unknown beta_schedule='{beta_schedule}'.")
        return np.asarray(betas, dtype=np.float64)

    def q_sample(self, x_start: Any, timestep: Any, noise: Optional[Any] = None) -> Any:
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            _extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: Any, timestep: Any, noise: Any) -> Any:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timestep, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, timestep, x_t.shape) * noise
        )

    def q_posterior(self, x_start: Any, x_t: Any, timestep: Any) -> tuple[Any, Any, Any]:
        mean = (
            _extract(self.posterior_mean_coef1, timestep, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, timestep, x_t.shape) * x_t
        )
        variance = _extract(self.posterior_variance, timestep, x_t.shape)
        log_variance = _extract(self.posterior_log_variance_clipped, timestep, x_t.shape)
        return mean, variance, log_variance

    def p_mean_variance(self, x: Any, cond: Any, timestep: Any, clip_denoised: bool = False) -> tuple[Any, Any, Any]:
        predicted_noise = self.denoise_fn(x, timestep, cond=cond)
        x_recon = self.predict_start_from_noise(x, timestep=timestep, noise=predicted_noise)
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x, timestep=timestep)

    def p_sample(self, x: Any, cond: Any, timestep: Any) -> Any:
        mean, _, log_variance = self.p_mean_variance(x=x, cond=cond, timestep=timestep)
        noise = torch.randn_like(x)
        nonzero_mask = (1.0 - (timestep == 0).float()).reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        return mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def loss(self, x: Any, cond: Any, scale: Optional[Any] = None) -> Any:
        if scale is not None:
            x = x / scale
        batch_size, sequence_length, target_dim = x.shape
        flat_x = x.reshape(batch_size * sequence_length, 1, target_dim)
        flat_cond = cond.reshape(batch_size * sequence_length, 1, cond.shape[-1])
        timestep = torch.randint(0, self.num_timesteps, (flat_x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(flat_x)
        noisy = self.q_sample(flat_x, timestep=timestep, noise=noise)
        predicted_noise = self.denoise_fn(noisy, timestep, cond=flat_cond)
        if self.loss_type == "l1":
            error = (predicted_noise - noise).abs()
        elif self.loss_type == "l2":
            error = (predicted_noise - noise).square()
        elif self.loss_type == "huber":
            error = F.smooth_l1_loss(predicted_noise, noise, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss_type='{self.loss_type}'.")
        return error.reshape(batch_size, sequence_length, target_dim).mean(dim=-1)

    def sample(self, cond: Any, scale: Optional[Any] = None) -> Any:
        shape = cond.shape[:-1] + (self.input_size,)
        x = torch.randn(shape, device=cond.device, dtype=cond.dtype)
        for step in reversed(range(self.num_timesteps)):
            timestep = torch.full((shape[0],), step, device=cond.device, dtype=torch.long)
            x = self.p_sample(x, cond=cond, timestep=timestep)
        if scale is not None:
            x = x * scale
        return x


class TimeGradNetwork(nn.Module if nn is not None else object):
    """RNN temporal conditioner plus conditional diffusion output."""

    def __init__(
        self,
        target_dim: int,
        time_feature_dim: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        dropout_rate: float,
        cardinality: Sequence[int],
        embedding_dimension: int,
        conditioning_length: int,
        diff_steps: int,
        loss_type: str,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        lags_seq: Sequence[int],
        scaling: bool,
        min_scale: float,
        residual_hidden: int = 64,
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
        self.embed_dim = int(embedding_dimension)
        self.scaling = bool(scaling)
        self.min_scale = float(min_scale)

        self.embed = nn.Embedding(self.target_dim, self.embed_dim)
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
        self.epsilon_theta = EpsilonTheta(
            target_dim=self.target_dim,
            cond_length=int(conditioning_length),
            time_emb_dim=16,
            residual_layers=int(residual_layers),
            residual_channels=int(residual_channels),
            dilation_cycle_length=int(dilation_cycle_length),
            residual_hidden=int(residual_hidden),
            max_steps=max(500, int(diff_steps)),
        )
        self.diffusion = GaussianDiffusion(
            denoise_fn=self.epsilon_theta,
            input_size=self.target_dim,
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            diff_steps=int(diff_steps),
            loss_type=str(loss_type),
            beta_schedule=str(beta_schedule),
        )
        self.cardinality = [int(item) for item in cardinality]

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
        target_embeddings = self._target_dimension_embeddings(past.shape[0], past.device)
        state = self._initial_state(past.shape[0], past.device)

        conditions = []
        targets = []
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
            conditions.append(self._condition_from_rnn(rnn_output[:, 0, :]))
            targets.append(past[:, step_index, :])
            weights.append(past_observed[:, step_index, :].min(dim=-1).values)

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
            weights.append(future_observed[:, step, :].min(dim=-1).values)
            history = torch.cat((history, future[:, step : step + 1, :]), dim=1)

        cond_seq = torch.stack(conditions, dim=1)
        target_seq = torch.stack(targets, dim=1)
        weight_seq = torch.stack(weights, dim=1)
        per_step_loss = self.diffusion.loss(target_seq, cond_seq, scale=scale)
        loss_sum = (per_step_loss * weight_seq).sum()
        weight_sum = weight_seq.sum()
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
            cond = self._condition_from_rnn(rnn_output)
            sample = self.diffusion.sample(cond=cond, scale=repeated_scale).squeeze(1)
            sample_steps.append(sample)
            history = torch.cat((history, sample.unsqueeze(1)), dim=1)

        flat_samples = torch.stack(sample_steps, dim=1)
        return flat_samples.reshape(past.shape[0], repeats, -1, self.target_dim)


class TimeGradForecastModel(BaseProbForecastModel):
    """TimeGrad model with FinProbTS fit/predict/save/load API."""

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
        conditioning_length: int = 100,
        diff_steps: int = 100,
        loss_type: str = "l2",
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
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
        hidden_size: Optional[int] = None,
        rnn_type: Optional[str] = None,
        dropout: Optional[float] = None,
        diffusion_steps: Optional[int] = None,
        step_embedding_dim: Optional[int] = None,
        denoiser_hidden_size: Optional[int] = None,
        residual_hidden: int = 64,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if hidden_size is not None:
            num_cells = int(hidden_size)
        if rnn_type is not None:
            cell_type = str(rnn_type)
        if dropout is not None:
            dropout_rate = float(dropout)
        if diffusion_steps is not None:
            diff_steps = int(diffusion_steps)
        if step_embedding_dim is not None:
            residual_hidden = max(int(residual_hidden), int(step_embedding_dim) * 4)
        if denoiser_hidden_size is not None:
            residual_hidden = int(denoiser_hidden_size)
        if learning_rate is not None:
            lr = float(learning_rate)
        if optim_kwargs:
            lr = float(optim_kwargs.get("lr", lr))
            weight_decay = float(optim_kwargs.get("weight_decay", weight_decay))
            patience = optim_kwargs.get("patience", patience)
        if time_features is not None:
            raise NotImplementedError("Custom TimeGrad time_features are not wired into FinProbTS yet.")
        if bool(pick_incomplete):
            raise NotImplementedError("TimeGrad pick_incomplete=True is an upstream sampler option; FinProbTS uses complete rolling windows.")

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
        self.conditioning_length = int(conditioning_length)
        self.diff_steps = int(diff_steps)
        self.diffusion_steps = self.diff_steps
        self.loss_type = str(loss_type)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = str(beta_schedule)
        self.residual_layers = int(residual_layers)
        self.residual_channels = int(residual_channels)
        self.dilation_cycle_length = int(dilation_cycle_length)
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
        self.residual_hidden = int(residual_hidden)

        self._network: Optional[TimeGradNetwork] = None
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
            "conditioning_length": self.conditioning_length,
            "diff_steps": self.diff_steps,
            "loss_type": self.loss_type,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "residual_layers": self.residual_layers,
            "residual_channels": self.residual_channels,
            "dilation_cycle_length": self.dilation_cycle_length,
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
            "residual_hidden": self.residual_hidden,
        }

    def _validate_data(self, data: RollingWindowDataset) -> None:
        if self.context_length is not None and data.context_length != self.context_length:
            raise ValueError(
                "TimeGrad context_length is controlled by the FinProbTS task window. "
                f"Got model context_length={self.context_length} but data has context_length={data.context_length}."
            )
        if self.prediction_length is not None and data.prediction_length != self.prediction_length:
            raise ValueError(
                "TimeGrad prediction_length is controlled by the FinProbTS task window. "
                f"Got model prediction_length={self.prediction_length} but data has prediction_length={data.prediction_length}."
            )
        if self.target_dim is not None and data.num_assets != self.target_dim:
            raise ValueError(f"TimeGrad target_dim={self.target_dim} does not match data num_assets={data.num_assets}.")
        if data.context_length < max(self.lags_seq):
            raise ValueError(
                f"TimeGrad requires context_length >= max(lags_seq). "
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
                f"TimeGrad input_size={self._effective_input_size}."
            )
        self._device = resolve_torch_device(self.device_name)
        self._network = TimeGradNetwork(
            target_dim=num_assets,
            time_feature_dim=time_feature_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            conditioning_length=self.conditioning_length,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            min_scale=self.min_scale,
            residual_hidden=self.residual_hidden,
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
                print(f"TimeGrad epoch {epoch}/{self.max_epochs}: train={train_loss:.6f} val={val_loss:.6f}")
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
                "model_name": "timegrad",
                "implementation": "finprobts_native_timegrad",
                "reference": "PyTorchTS TimeGradEstimator",
                "seed": self.seed,
                "diff_steps": self.diff_steps,
                "diffusion_steps": self.diffusion_steps,
                "loss_type": self.loss_type,
                "beta_schedule": self.beta_schedule,
                "beta_end": self.beta_end,
                "num_cells": self.num_cells,
                "num_layers": self.num_layers,
                "residual_layers": self.residual_layers,
                "residual_channels": self.residual_channels,
                "conditioning_length": self.conditioning_length,
                "lags_seq": list(self.lags_seq),
                "model_internal_scaling": self.scaling,
                "effective_input_size": self._effective_input_size,
                "num_parallel_samples_default": self.num_parallel_samples,
                "training_history": list(self.training_history),
            },
        )

    def save(self, path: str) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("Cannot save an unfitted TimeGradForecastModel.")
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
    def load(cls, path: str) -> "TimeGradForecastModel":
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
