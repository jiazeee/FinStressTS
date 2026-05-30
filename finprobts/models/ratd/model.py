"""Native RATD forecaster for the FinProbTS rolling-window contract.

RATD frames forecasting as conditional imputation: history points are observed,
future points are masked, and a diffusion model denoises the full
``history + horizon`` panel while being guided by retrieved reference futures.

This implementation mirrors the public RATD/CSDI design with pure PyTorch:
sinusoidal time embeddings, asset embeddings, conditional masks, residual
time/feature transformer blocks, diffusion-step embeddings, and
reference-modulated cross-asset attention.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from finprobts.data.schema import RollingWindowDataset
from finprobts.models.base import BaseProbForecastModel, ForecastResult
from finprobts.models.torch_utils import (
    TorchStandardScaler,
    iter_torch_batches,
    make_torch_data_loader,
    make_window_arrays,
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


def _make_beta_schedule(schedule: str, beta_start: float, beta_end: float, num_steps: int) -> Any:
    if str(schedule).lower() == "quad":
        return torch.linspace(float(beta_start) ** 0.5, float(beta_end) ** 0.5, int(num_steps)).square()
    if str(schedule).lower() == "linear":
        return torch.linspace(float(beta_start), float(beta_end), int(num_steps))
    raise ValueError("schedule must be 'quad' or 'linear'.")


def _conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int) -> Any:
    require_torch()
    layer = nn.Conv1d(int(in_channels), int(out_channels), int(kernel_size))
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module if nn is not None else object):
    """Sinusoidal diffusion-step embedding used by CSDI/RATD blocks."""

    def __init__(self, num_steps: int, embedding_dim: int = 128, projection_dim: Optional[int] = None) -> None:
        require_torch()
        super().__init__()
        projection_dim = int(embedding_dim if projection_dim is None else projection_dim)
        self.register_buffer(
            "embedding",
            self._build_embedding(int(num_steps), int(embedding_dim) // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(int(embedding_dim), projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    @staticmethod
    def _build_embedding(num_steps: int, dim: int) -> Any:
        steps = torch.arange(int(num_steps), dtype=torch.float32).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(int(dim), dtype=torch.float32) / max(int(dim) - 1, 1) * 4.0)
        table = steps * frequencies.unsqueeze(0)
        return torch.cat((torch.sin(table), torch.cos(table)), dim=1)

    def forward(self, diffusion_step: Any) -> Any:
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))


class ReferenceModulatedCrossAttention(nn.Module if nn is not None else object):
    """RATD-style cross-asset attention using retrieved future references."""

    def __init__(
        self,
        sequence_dim: int,
        reference_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        require_torch()
        super().__init__()
        self.sequence_dim = int(sequence_dim)
        self.reference_dim = int(reference_dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.inner_dim = self.heads * self.dim_head
        self.scale = self.dim_head ** -0.5
        self.y_to_q = nn.Linear(self.sequence_dim, self.inner_dim, bias=False)
        self.cond_to_k = nn.Linear(2 * self.sequence_dim + self.reference_dim, self.inner_dim, bias=False)
        self.ref_to_v = nn.Linear(self.sequence_dim + self.reference_dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, self.sequence_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: Any, cond_info: Any, reference: Any) -> Any:
        batch_size, channels, num_assets, sequence_length = x.shape
        if reference.shape[-1] != self.reference_dim:
            raise ValueError(
                f"reference last dimension={reference.shape[-1]} does not match expected {self.reference_dim}."
            )
        x_flat = x.reshape(batch_size * channels, num_assets, sequence_length)
        cond_flat = cond_info.reshape(batch_size * channels, num_assets, sequence_length)
        ref_flat = reference.unsqueeze(1).expand(-1, channels, -1, -1)
        ref_flat = ref_flat.reshape(batch_size * channels, num_assets, self.reference_dim)

        q = self.y_to_q(x_flat)
        k = self.cond_to_k(torch.cat((x_flat, cond_flat, ref_flat), dim=-1))
        v = self.ref_to_v(torch.cat((x_flat, ref_flat), dim=-1))
        q = q.reshape(batch_size * channels, num_assets, self.heads, self.dim_head).transpose(1, 2)
        k = k.reshape(batch_size * channels, num_assets, self.heads, self.dim_head).transpose(1, 2)
        v = v.reshape(batch_size * channels, num_assets, self.heads, self.dim_head).transpose(1, 2)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = self.dropout(torch.softmax(attention, dim=-1))
        out = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size * channels, num_assets, self.inner_dim)
        out = self.to_out(out)
        return out.reshape(batch_size, channels, num_assets, sequence_length)


class RATDResidualBlock(nn.Module if nn is not None else object):
    """Residual CSDI block with time/feature transformers and reference fusion."""

    def __init__(
        self,
        side_dim: int,
        sequence_length: int,
        reference_length: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        use_reference: bool,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        self.channels = int(channels)
        self.use_reference = bool(use_reference)
        self.diffusion_projection = nn.Linear(int(diffusion_embedding_dim), self.channels)
        self.cond_projection = _conv1d_with_init(int(side_dim), self.channels, 1)
        self.mid_projection = _conv1d_with_init(self.channels, 2 * self.channels, 1)
        self.output_projection = _conv1d_with_init(self.channels, 2 * self.channels, 1)
        self.time_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.channels,
                nhead=int(nheads),
                dim_feedforward=64,
                dropout=float(dropout),
                activation="gelu",
            ),
            num_layers=1,
        )
        self.feature_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.channels,
                nhead=int(nheads),
                dim_feedforward=64,
                dropout=float(dropout),
                activation="gelu",
            ),
            num_layers=1,
        )
        self.reference_attention = (
            ReferenceModulatedCrossAttention(
                sequence_dim=int(sequence_length),
                reference_dim=int(reference_length),
                heads=8,
                dim_head=64,
                dropout=0.0,
            )
            if self.use_reference
            else None
        )

    def _forward_time(self, y: Any, base_shape: tuple[int, int, int, int]) -> Any:
        batch_size, channels, num_assets, sequence_length = base_shape
        if sequence_length == 1:
            return y
        y = y.reshape(batch_size, channels, num_assets, sequence_length)
        y = y.permute(3, 0, 2, 1).reshape(sequence_length, batch_size * num_assets, channels)
        y = self.time_layer(y)
        y = y.reshape(sequence_length, batch_size, num_assets, channels)
        return y.permute(1, 3, 2, 0).reshape(batch_size, channels, num_assets * sequence_length)

    def _forward_feature(self, y: Any, base_shape: tuple[int, int, int, int]) -> Any:
        batch_size, channels, num_assets, sequence_length = base_shape
        if num_assets == 1:
            return y
        y = y.reshape(batch_size, channels, num_assets, sequence_length)
        y = y.permute(2, 0, 3, 1).reshape(num_assets, batch_size * sequence_length, channels)
        y = self.feature_layer(y)
        y = y.reshape(num_assets, batch_size, sequence_length, channels)
        return y.permute(1, 3, 0, 2).reshape(batch_size, channels, num_assets * sequence_length)

    def forward(self, x: Any, side_info: Any, diffusion_emb: Any, reference: Optional[Any]) -> tuple[Any, Any]:
        batch_size, channels, num_assets, sequence_length = x.shape
        base_shape = (batch_size, channels, num_assets, sequence_length)
        x_flat = x.reshape(batch_size, channels, num_assets * sequence_length)
        y = x_flat + self.diffusion_projection(diffusion_emb).unsqueeze(-1)

        cond = self.cond_projection(side_info.reshape(batch_size, side_info.shape[1], num_assets * sequence_length))
        if self.reference_attention is not None and reference is not None:
            cond = self.reference_attention(
                y.reshape(base_shape),
                cond.reshape(base_shape),
                reference,
            ).reshape(batch_size, channels, num_assets * sequence_length)

        y = y + cond
        y = self._forward_time(y, base_shape)
        y = self._forward_feature(y, base_shape)
        y = self.mid_projection(y)
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class RATDDiffusionNetwork(nn.Module if nn is not None else object):
    """CSDI/RATD diffusion denoiser for panels shaped ``[B, D, L]``."""

    def __init__(
        self,
        input_dim: int,
        side_dim: int,
        sequence_length: int,
        reference_length: int,
        layers: int,
        channels: int,
        nheads: int,
        diffusion_embedding_dim: int,
        use_reference: bool,
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        self.channels = int(channels)
        self.input_projection = _conv1d_with_init(int(input_dim), self.channels, 1)
        self.output_projection1 = _conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = _conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=1,  # replaced by parent with direct table indexing through max step buffers
            embedding_dim=int(diffusion_embedding_dim),
        )
        self.residual_layers = nn.ModuleList(
            [
                RATDResidualBlock(
                    side_dim=int(side_dim),
                    sequence_length=int(sequence_length),
                    reference_length=int(reference_length),
                    channels=self.channels,
                    diffusion_embedding_dim=int(diffusion_embedding_dim),
                    nheads=int(nheads),
                    use_reference=bool(use_reference),
                    dropout=float(dropout),
                )
                for _ in range(int(layers))
            ]
        )

    def replace_diffusion_embedding(self, num_steps: int, diffusion_embedding_dim: int) -> None:
        self.diffusion_embedding = DiffusionEmbedding(int(num_steps), int(diffusion_embedding_dim))

    def forward(self, x: Any, side_info: Any, diffusion_step: Any, reference: Optional[Any]) -> Any:
        batch_size, input_dim, num_assets, sequence_length = x.shape
        y = x.reshape(batch_size, input_dim, num_assets * sequence_length)
        y = F.relu(self.input_projection(y))
        y = y.reshape(batch_size, self.channels, num_assets, sequence_length)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            y, skip_connection = layer(y, side_info, diffusion_emb, reference)
            skip.append(skip_connection)

        y = torch.stack(skip, dim=0).sum(dim=0) / math.sqrt(len(skip))
        y = y.reshape(batch_size, self.channels, num_assets * sequence_length)
        y = F.relu(self.output_projection1(y))
        y = self.output_projection2(y)
        return y.reshape(batch_size, num_assets, sequence_length)


class RATDNetwork(nn.Module if nn is not None else object):
    """Reference-guided diffusion model for forecasting-as-imputation."""

    def __init__(
        self,
        num_assets: int,
        context_length: int,
        prediction_length: int,
        layers: int,
        channels: int,
        nheads: int,
        diffusion_embedding_dim: int,
        beta_start: float,
        beta_end: float,
        diffusion_steps: int,
        schedule: str,
        time_embedding_dim: int,
        feature_embedding_dim: int,
        is_unconditional: bool,
        use_reference: bool,
        retrieval_k: int,
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        self.num_assets = int(num_assets)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.sequence_length = self.context_length + self.prediction_length
        self.diffusion_steps = int(diffusion_steps)
        self.time_embedding_dim = int(time_embedding_dim)
        self.feature_embedding_dim = int(feature_embedding_dim)
        self.is_unconditional = bool(is_unconditional)
        self.use_reference = bool(use_reference)
        self.retrieval_k = int(retrieval_k)
        self.side_dim = self.time_embedding_dim + self.feature_embedding_dim + (0 if self.is_unconditional else 1)
        self.embed_layer = nn.Embedding(self.num_assets, self.feature_embedding_dim)
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = RATDDiffusionNetwork(
            input_dim=input_dim,
            side_dim=self.side_dim,
            sequence_length=self.sequence_length,
            reference_length=max(1, self.retrieval_k * self.prediction_length),
            layers=int(layers),
            channels=int(channels),
            nheads=int(nheads),
            diffusion_embedding_dim=int(diffusion_embedding_dim),
            use_reference=bool(use_reference),
            dropout=float(dropout),
        )
        self.diffmodel.replace_diffusion_embedding(self.diffusion_steps, int(diffusion_embedding_dim))

        betas = _make_beta_schedule(schedule, beta_start, beta_end, self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def time_embedding(self, positions: Any) -> Any:
        dim = self.time_embedding_dim
        pe = torch.zeros(positions.shape[0], positions.shape[1], dim, device=positions.device)
        position = positions.unsqueeze(2)
        div_term = 1.0 / torch.pow(10000.0, torch.arange(0, dim, 2, device=positions.device).float() / max(dim, 1))
        angles = position * div_term
        pe[:, :, 0::2] = torch.sin(angles[:, :, : pe[:, :, 0::2].shape[-1]])
        pe[:, :, 1::2] = torch.cos(angles[:, :, : pe[:, :, 1::2].shape[-1]])
        return pe

    def get_side_info(self, timepoints: Any, cond_mask: Any) -> Any:
        batch_size, num_assets, sequence_length = cond_mask.shape
        time_embed = self.time_embedding(timepoints)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, num_assets, -1)
        feature_embed = self.embed_layer(torch.arange(num_assets, device=cond_mask.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, sequence_length, -1, -1)
        side_info = torch.cat((time_embed, feature_embed), dim=-1).permute(0, 3, 2, 1)
        if not self.is_unconditional:
            side_info = torch.cat((side_info, cond_mask.unsqueeze(1)), dim=1)
        return side_info

    def set_input_to_diffmodel(self, noisy_data: Any, observed_data: Any, cond_mask: Any) -> Any:
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1.0 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat((cond_obs, noisy_target), dim=1)

    def loss(self, batch: Dict[str, Any], reference: Optional[Any], validate_all_steps: bool = False) -> Any:
        observed_data, observed_mask, cond_mask, timepoints = make_ratd_batch_tensors(batch)
        side_info = self.get_side_info(timepoints, cond_mask)
        if validate_all_steps:
            losses = [
                self._loss_for_step(observed_data, observed_mask, cond_mask, side_info, reference, step=t)
                for t in range(self.diffusion_steps)
            ]
            return torch.stack(losses).mean()
        return self._loss_for_step(observed_data, observed_mask, cond_mask, side_info, reference, step=None)

    def _loss_for_step(
        self,
        observed_data: Any,
        observed_mask: Any,
        cond_mask: Any,
        side_info: Any,
        reference: Optional[Any],
        step: Optional[int],
    ) -> Any:
        batch_size = observed_data.shape[0]
        if step is None:
            timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=observed_data.device)
        else:
            timestep = torch.full((batch_size,), int(step), device=observed_data.device, dtype=torch.long)
        alpha_bar = self.alpha_bars[timestep].reshape(batch_size, 1, 1)
        noise = torch.randn_like(observed_data)
        noisy_data = torch.sqrt(alpha_bar) * observed_data + torch.sqrt(1.0 - alpha_bar) * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, timestep, reference=reference)
        target_mask = (observed_mask - cond_mask).clamp_min(0.0)
        residual = (noise - predicted) * target_mask
        denom = target_mask.sum().clamp_min(1.0)
        return residual.square().sum() / denom

    def sample(self, batch: Dict[str, Any], reference: Optional[Any], num_samples: int) -> Any:
        observed_data, _, cond_mask, timepoints = make_ratd_batch_tensors(batch, zero_future=True)
        batch_size, num_assets, sequence_length = observed_data.shape
        n_samples = int(num_samples)
        repeated_observed = observed_data.repeat_interleave(n_samples, dim=0)
        repeated_cond_mask = cond_mask.repeat_interleave(n_samples, dim=0)
        repeated_timepoints = timepoints.repeat_interleave(n_samples, dim=0)
        repeated_reference = None if reference is None else reference.repeat_interleave(n_samples, dim=0)
        side_info = self.get_side_info(repeated_timepoints, repeated_cond_mask)
        current_sample = torch.randn_like(repeated_observed)

        for step in reversed(range(self.diffusion_steps)):
            timestep = torch.full((current_sample.shape[0],), step, device=current_sample.device, dtype=torch.long)
            diff_input = self.set_input_to_diffmodel(current_sample, repeated_observed, repeated_cond_mask)
            predicted = self.diffmodel(diff_input, side_info, timestep, reference=repeated_reference)
            coeff1 = 1.0 / torch.sqrt(self.alphas[step])
            coeff2 = (1.0 - self.alphas[step]) / torch.sqrt(1.0 - self.alpha_bars[step])
            current_sample = coeff1 * (current_sample - coeff2 * predicted)
            if step > 0:
                sigma = torch.sqrt((1.0 - self.alpha_bars[step - 1]) / (1.0 - self.alpha_bars[step]) * self.betas[step])
                current_sample = current_sample + sigma * torch.randn_like(current_sample)

        future = current_sample[:, :, self.context_length :]
        future = future.reshape(batch_size, n_samples, num_assets, self.prediction_length)
        return future.permute(0, 1, 3, 2)


def make_ratd_batch_tensors(batch: Dict[str, Any], zero_future: bool = False) -> tuple[Any, Any, Any, Any]:
    """Build RATD observed data, mask, condition mask, and timepoints."""

    past = batch["past_target"]
    future = torch.zeros_like(batch["future_target"]) if zero_future else batch["future_target"]
    past_observed = batch["past_observed_values"]
    future_observed = batch["future_observed_values"]
    observed = torch.cat((past, future), dim=1).permute(0, 2, 1)
    observed_mask = torch.cat((past_observed, future_observed), dim=1).permute(0, 2, 1)
    cond_mask = torch.cat((past_observed, torch.zeros_like(future_observed)), dim=1).permute(0, 2, 1)
    timepoints = torch.arange(observed.shape[-1], device=observed.device, dtype=observed.dtype)
    timepoints = timepoints.unsqueeze(0).expand(observed.shape[0], -1)
    return observed, observed_mask, cond_mask, timepoints


class RATDForecastModel(BaseProbForecastModel):
    """Native Retrieval-Augmented Time series Diffusion model."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        freq: Optional[str] = None,
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        context_length: Optional[int] = None,
        layers: int = 4,
        channels: int = 64,
        nheads: int = 8,
        diffusion_embedding_dim: int = 128,
        beta_start: float = 1e-4,
        beta_end: float = 0.5,
        diffusion_steps: int = 50,
        num_steps: Optional[int] = None,
        schedule: str = "quad",
        time_embedding_dim: int = 128,
        feature_embedding_dim: int = 16,
        timeemb: Optional[int] = None,
        featureemb: Optional[int] = None,
        is_unconditional: bool = False,
        target_strategy: str = "test",
        num_sample_features: int = 64,
        use_reference: bool = True,
        retrieval_k: int = 3,
        retrieval_metric: str = "cosine",
        retrieval_exclude_self: bool = True,
        batch_size: int = 1,
        max_epochs: int = 10,
        learning_rate: float = 3e-4,
        lr: Optional[float] = None,
        weight_decay: float = 1e-6,
        gradient_clip_val: Optional[float] = 1.0,
        patience: Optional[int] = 10,
        validation_all_timesteps: bool = False,
        device: str = "auto",
        seed: Optional[int] = None,
        scaling: bool = True,
        scaler_min_std: float = 1e-6,
        dropout: float = 0.1,
        verbose: bool = False,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        rnn_type: Optional[str] = None,
        denoiser_hidden_size: Optional[int] = None,
        reference_hidden_size: Optional[int] = None,
        step_embedding_dim: Optional[int] = None,
        diff_steps: Optional[int] = None,
        residual_channels: Optional[int] = None,
        reference_dim: Optional[int] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if num_steps is not None:
            diffusion_steps = int(num_steps)
        if diff_steps is not None:
            diffusion_steps = int(diff_steps)
        if timeemb is not None:
            time_embedding_dim = int(timeemb)
        if featureemb is not None:
            feature_embedding_dim = int(featureemb)
        if hidden_size is not None:
            channels = int(hidden_size)
        if residual_channels is not None:
            channels = int(residual_channels)
        if num_layers is not None:
            layers = int(num_layers)
        if step_embedding_dim is not None:
            diffusion_embedding_dim = int(step_embedding_dim)
        if lr is not None:
            learning_rate = float(lr)
        if optim_kwargs:
            learning_rate = float(optim_kwargs.get("lr", learning_rate))
            weight_decay = float(optim_kwargs.get("weight_decay", weight_decay))
            patience = optim_kwargs.get("patience", patience)

        del rnn_type, denoiser_hidden_size, reference_hidden_size, reference_dim

        self.input_size = None if input_size is None else int(input_size)
        self.freq = None if freq is None else str(freq)
        self.prediction_length = None if prediction_length is None else int(prediction_length)
        self.target_dim = None if target_dim is None else int(target_dim)
        self.context_length = None if context_length is None else int(context_length)
        self.layers = int(layers)
        self.channels = int(channels)
        self.nheads = int(nheads)
        if self.channels % self.nheads != 0:
            raise ValueError("channels must be divisible by nheads for RATD transformer blocks.")
        self.diffusion_embedding_dim = int(diffusion_embedding_dim)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.diffusion_steps = int(diffusion_steps)
        self.schedule = str(schedule)
        self.time_embedding_dim = int(time_embedding_dim)
        self.feature_embedding_dim = int(feature_embedding_dim)
        self.is_unconditional = bool(is_unconditional)
        self.target_strategy = str(target_strategy)
        if self.target_strategy != "test":
            raise NotImplementedError("FinProbTS RATD currently uses the upstream forecasting target_strategy='test'.")
        self.num_sample_features = int(num_sample_features)
        self.use_reference = bool(use_reference)
        self.retrieval_k = max(1, int(retrieval_k))
        self.retrieval_metric = str(retrieval_metric).lower()
        if self.retrieval_metric not in {"cosine", "euclidean"}:
            raise ValueError("retrieval_metric must be 'cosine' or 'euclidean'.")
        self.retrieval_exclude_self = bool(retrieval_exclude_self)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.lr = self.learning_rate
        self.weight_decay = float(weight_decay)
        self.gradient_clip_val = None if gradient_clip_val is None else float(gradient_clip_val)
        self.patience = None if patience is None else int(patience)
        self.validation_all_timesteps = bool(validation_all_timesteps)
        self.device_name = device
        self.seed = seed
        self.scaling = bool(scaling)
        self.scaler_min_std = float(scaler_min_std)
        self.dropout = float(dropout)
        self.verbose = bool(verbose)

        self._network: Optional[RATDNetwork] = None
        self._device = None
        self._scaler: Optional[TorchStandardScaler] = None
        self._num_assets: Optional[int] = None
        self._fit_context_length: Optional[int] = None
        self._fit_prediction_length: Optional[int] = None
        self._asset_ids: Optional[list[str]] = None
        self._memory_contexts = None
        self._memory_targets = None
        self._is_fitted = False
        self.training_history: list[Dict[str, float]] = []

    def _init_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "freq": self.freq,
            "prediction_length": self.prediction_length,
            "target_dim": self.target_dim,
            "context_length": self.context_length,
            "layers": self.layers,
            "channels": self.channels,
            "nheads": self.nheads,
            "diffusion_embedding_dim": self.diffusion_embedding_dim,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "diffusion_steps": self.diffusion_steps,
            "schedule": self.schedule,
            "time_embedding_dim": self.time_embedding_dim,
            "feature_embedding_dim": self.feature_embedding_dim,
            "is_unconditional": self.is_unconditional,
            "target_strategy": self.target_strategy,
            "num_sample_features": self.num_sample_features,
            "use_reference": self.use_reference,
            "retrieval_k": self.retrieval_k,
            "retrieval_metric": self.retrieval_metric,
            "retrieval_exclude_self": self.retrieval_exclude_self,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_val": self.gradient_clip_val,
            "patience": self.patience,
            "validation_all_timesteps": self.validation_all_timesteps,
            "device": self.device_name,
            "seed": self.seed,
            "scaling": self.scaling,
            "scaler_min_std": self.scaler_min_std,
            "dropout": self.dropout,
            "verbose": self.verbose,
        }

    def _validate_data(self, data: RollingWindowDataset) -> None:
        if self.context_length is not None and data.context_length != self.context_length:
            raise ValueError(
                f"RATD context_length={self.context_length} does not match data context_length={data.context_length}."
            )
        if self.prediction_length is not None and data.prediction_length != self.prediction_length:
            raise ValueError(
                f"RATD prediction_length={self.prediction_length} does not match data prediction_length={data.prediction_length}."
            )
        if self.target_dim is not None and data.num_assets != self.target_dim:
            raise ValueError(f"RATD target_dim={self.target_dim} does not match data num_assets={data.num_assets}.")

    def _build_network(self, num_assets: int, context_length: int, prediction_length: int) -> None:
        self._num_assets = int(num_assets)
        self._fit_context_length = int(context_length)
        self._fit_prediction_length = int(prediction_length)
        effective_input_size = 2 if not self.is_unconditional else 1
        if self.input_size is not None and self.input_size != effective_input_size:
            raise ValueError(f"Configured input_size={self.input_size} does not match RATD input_dim={effective_input_size}.")
        self._device = resolve_torch_device(self.device_name)
        self._network = RATDNetwork(
            num_assets=num_assets,
            context_length=context_length,
            prediction_length=prediction_length,
            layers=self.layers,
            channels=self.channels,
            nheads=self.nheads,
            diffusion_embedding_dim=self.diffusion_embedding_dim,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            diffusion_steps=self.diffusion_steps,
            schedule=self.schedule,
            time_embedding_dim=self.time_embedding_dim,
            feature_embedding_dim=self.feature_embedding_dim,
            is_unconditional=self.is_unconditional,
            use_reference=self.use_reference,
            retrieval_k=self.retrieval_k,
            dropout=self.dropout,
        ).to(self._device)

    def _build_memory(self, train_data: RollingWindowDataset) -> None:
        assert self._device is not None
        arrays = make_window_arrays(train_data, scaler=self._scaler, include_time_features=False)
        contexts = arrays["past_target"].reshape(len(train_data), -1)
        targets = arrays["future_target"]
        context_norm = np.linalg.norm(contexts, axis=1, keepdims=True) + 1e-8
        contexts = contexts / context_norm if self.retrieval_metric == "cosine" else contexts
        self._memory_contexts = torch.as_tensor(contexts, dtype=torch.float32, device=self._device)
        self._memory_targets = torch.as_tensor(targets, dtype=torch.float32, device=self._device)

    def _reference_from_past(
        self,
        past_target: Any,
        window_index: Optional[Any] = None,
        exclude_by_index: bool = False,
    ) -> Optional[Any]:
        if not self.use_reference:
            return None
        if self._memory_contexts is None or self._memory_targets is None:
            raise RuntimeError("RATD retrieval memory has not been built.")
        flat = past_target.reshape(past_target.shape[0], -1)
        if flat.shape[1] != self._memory_contexts.shape[1]:
            raise ValueError("past_target context shape does not match the fitted RATD retrieval memory.")
        k = min(self.retrieval_k, int(self._memory_targets.shape[0]))
        num_memory = int(self._memory_targets.shape[0])
        candidate_k = min(num_memory, k + (1 if self.retrieval_exclude_self else 0))
        use_index_exclusion = self.retrieval_exclude_self and exclude_by_index and window_index is not None
        with torch.no_grad():
            if self.retrieval_metric == "cosine":
                query = F.normalize(flat, dim=-1)
                scores = query @ self._memory_contexts.transpose(0, 1)
                if use_index_exclusion:
                    self._suppress_indexed_self_matches(scores, window_index, fill_value=-float("inf"))
                    indices = torch.topk(scores, k=k, dim=-1).indices
                else:
                    indices = torch.topk(scores, k=candidate_k, dim=-1).indices
                if self.retrieval_exclude_self and not use_index_exclusion and candidate_k > k:
                    gathered = torch.gather(scores, 1, indices)
                    drop_first = gathered[:, 0] > 1.0 - 1e-6
                    indices = torch.where(drop_first[:, None], indices[:, 1 : k + 1], indices[:, :k])
                else:
                    indices = indices[:, :k]
            else:
                distances = torch.cdist(flat, self._memory_contexts)
                if use_index_exclusion:
                    self._suppress_indexed_self_matches(distances, window_index, fill_value=float("inf"))
                    indices = torch.topk(distances, k=k, largest=False, dim=-1).indices
                else:
                    indices = torch.topk(distances, k=candidate_k, largest=False, dim=-1).indices
                if self.retrieval_exclude_self and not use_index_exclusion and candidate_k > k:
                    gathered = torch.gather(distances, 1, indices)
                    drop_first = gathered[:, 0] < 1e-8
                    indices = torch.where(drop_first[:, None], indices[:, 1 : k + 1], indices[:, :k])
                else:
                    indices = indices[:, :k]
            refs = self._memory_targets[indices]
            if refs.shape[1] < self.retrieval_k:
                pad = refs[:, -1:, :, :].expand(-1, self.retrieval_k - refs.shape[1], -1, -1)
                refs = torch.cat((refs, pad), dim=1)
            return refs.permute(0, 3, 1, 2).reshape(past_target.shape[0], past_target.shape[-1], -1)

    def _suppress_indexed_self_matches(self, matrix: Any, window_index: Any, fill_value: float) -> None:
        """Mask exact same-window retrieval rows, matching RATD precomputed-index exclusion."""

        index = window_index.to(matrix.device).long().reshape(-1)
        valid = (index >= 0) & (index < matrix.shape[1])
        if bool(valid.any().detach().cpu()):
            rows = torch.arange(matrix.shape[0], device=matrix.device)[valid]
            matrix[rows, index[valid]] = fill_value

    def _make_loader(self, data: RollingWindowDataset, shuffle: bool) -> Any:
        return make_torch_data_loader(data, self.batch_size, shuffle, self._scaler, include_time_features=False)

    def _evaluate(self, loader: Any) -> float:
        assert self._network is not None and self._device is not None
        self._network.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                reference = self._reference_from_past(batch["past_target"])
                loss = self._network.loss(batch, reference, validate_all_steps=self.validation_all_timesteps)
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
        self._scaler = TorchStandardScaler.fit(train_data.x_context, min_std=self.scaler_min_std) if self.scaling else None
        self._build_network(train_data.num_assets, train_data.context_length, train_data.prediction_length)
        self._build_memory(train_data)
        train_loader = self._make_loader(train_data, shuffle=True)
        val_loader = self._make_loader(val_data, shuffle=False) if val_data is not None and len(val_data) > 0 else None
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

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
                reference = self._reference_from_past(
                    batch["past_target"],
                    window_index=batch.get("window_index"),
                    exclude_by_index=True,
                )
                loss = self._network.loss(batch, reference, validate_all_steps=False)
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
                print(f"RATD epoch {epoch}/{self.max_epochs}: train={train_loss:.6f} val={val_loss:.6f}")
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
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self._num_assets != test_data.num_assets:
            raise ValueError("test_data num_assets does not match the fitted model.")
        if self._fit_context_length != test_data.context_length:
            raise ValueError("test_data context_length does not match the fitted RATD retrieval memory.")
        if self._fit_prediction_length != test_data.prediction_length:
            raise ValueError("test_data prediction_length does not match the fitted RATD network.")

        loader = self._make_loader(test_data, shuffle=False)
        chunks = []
        self._network.eval()
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                reference = self._reference_from_past(batch["past_target"])
                chunks.append(self._network.sample(batch, reference, int(num_samples)).cpu().numpy())
        samples = np.concatenate(chunks, axis=0)
        if self._scaler is not None:
            samples = self._scaler.inverse_transform_array(samples)
        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "ratd",
                "implementation": "finprobts_native_ratd",
                "reference": "stanliu96/RATD",
                "seed": self.seed,
                "layers": self.layers,
                "channels": self.channels,
                "nheads": self.nheads,
                "diffusion_steps": self.diffusion_steps,
                "schedule": self.schedule,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "time_embedding_dim": self.time_embedding_dim,
                "feature_embedding_dim": self.feature_embedding_dim,
                "use_reference": self.use_reference,
                "retrieval_k": self.retrieval_k,
                "retrieval_metric": self.retrieval_metric,
                "retrieval_exclude_self": self.retrieval_exclude_self,
                "model_internal_scaling": self.scaling,
                "training_history": list(self.training_history),
            },
        )

    def save(self, path: str) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("Cannot save an unfitted RATDForecastModel.")
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "init_params": self._init_params(),
                "model_state": self._network.state_dict(),
                "num_assets": self._num_assets,
                "fit_context_length": self._fit_context_length,
                "fit_prediction_length": self._fit_prediction_length,
                "asset_ids": self._asset_ids,
                "memory_contexts": None if self._memory_contexts is None else self._memory_contexts.detach().cpu(),
                "memory_targets": None if self._memory_targets is None else self._memory_targets.detach().cpu(),
                "scaler_state": self._scaler.state_dict() if self._scaler is not None else None,
                "training_history": self.training_history,
                "is_fitted": self._is_fitted,
            },
            output_dir / "model.pt",
        )

    @classmethod
    def load(cls, path: str) -> "RATDForecastModel":
        require_torch()
        try:
            payload = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(Path(path) / "model.pt", map_location="cpu")
        model = cls(**payload["init_params"])
        model._asset_ids = [str(item) for item in payload.get("asset_ids", [])]
        model._build_network(
            int(payload["num_assets"]),
            int(payload["fit_context_length"]),
            int(payload["fit_prediction_length"]),
        )
        model._network.load_state_dict(payload["model_state"])
        memory_contexts = payload.get("memory_contexts")
        memory_targets = payload.get("memory_targets")
        model._memory_contexts = None if memory_contexts is None else memory_contexts.to(model._device)
        model._memory_targets = None if memory_targets is None else memory_targets.to(model._device)
        model._fit_context_length = payload.get("fit_context_length")
        model._fit_prediction_length = payload.get("fit_prediction_length")
        scaler_state = payload.get("scaler_state")
        model._scaler = TorchStandardScaler.from_state_dict(scaler_state) if scaler_state is not None else None
        model.training_history = list(payload.get("training_history", []))
        model._is_fitted = bool(payload.get("is_fitted", True))
        return model
