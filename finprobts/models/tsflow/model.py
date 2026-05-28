"""Native TSFlow forecaster for the FinProbTS rolling-window contract.

TSFlow is a conditional flow-matching model for probabilistic time-series
forecasting. Its key idea is to sample the source path from a
data-dependent Gaussian-process prior instead of a fixed isotropic Gaussian,
then learn a vector field from that prior sample to the observed future.

This native implementation keeps the FinProbTS interface while following the
public TSFlow design: GP-regression priors, conditional flow matching, lag and
observation-mask features, sinusoidal flow-time embeddings, residual temporal
and cross-asset sequence blocks, optional EMA weights, and Euler sampling.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from finprobts.data.schema import RollingWindowDataset
from finprobts.models.base import BaseProbForecastModel, ForecastResult
from finprobts.models.torch_utils import (
    TorchStandardScaler,
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


def _season_length(freq: Optional[str]) -> int:
    if freq is None:
        return 30
    normalized = str(freq).upper()
    return {
        "H": 24,
        "D": 30,
        "1D": 30,
        "B": 30,
        "W": 7,
        "M": 12,
    }.get(normalized, 30)


def _default_lags_for_freq(freq: Optional[str]) -> list[int]:
    normalized = "B" if freq is None else str(freq).upper()
    if normalized == "H":
        return [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    if normalized in {"D", "1D"}:
        return [30 * i for i in [1, 2, 3, 4, 5, 6, 7]]
    if normalized == "B":
        return [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    return [1, 2, 7, 14, 30]


def _normalize_lags(lags_seq: Optional[Sequence[int]], freq: Optional[str], context_length: int) -> list[int]:
    raw = list(lags_seq) if lags_seq is not None else _default_lags_for_freq(freq)
    lags = sorted({int(lag) for lag in raw if int(lag) > 0 and int(lag) < int(context_length)})
    if not lags:
        lags = [1]
    return lags


class SinusoidalPositionEmbeddings(nn.Module if nn is not None else object):
    """Sinusoidal embedding for flow time ``t``."""

    def __init__(self, dim: int) -> None:
        require_torch()
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: Any) -> Any:
        half_dim = max(self.dim // 2, 1)
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=time.device, dtype=time.dtype) * -scale)
        angles = time[:, None] * frequencies[None, :]
        embeddings = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embeddings.shape[-1] < self.dim:
            embeddings = F.pad(embeddings, (0, self.dim - embeddings.shape[-1]))
        return embeddings[:, : self.dim]


class GaussianProcessPrior(nn.Module if nn is not None else object):
    """Conditional Gaussian-process prior used as TSFlow source distribution."""

    def __init__(
        self,
        kernel: str = "ou",
        context_freqs: int = 14,
        prediction_length: int = 1,
        freq: int = 30,
        gamma: float = 1.0,
        iso: float = 1e-1,
    ) -> None:
        require_torch()
        super().__init__()
        self.kernel = str(kernel).lower()
        self.context_freqs = int(context_freqs)
        self.prediction_length = int(prediction_length)
        self.prior_context_length = max(1, self.context_freqs * self.prediction_length)
        self.freq = int(freq)
        self.gamma = float(gamma)
        self.iso = float(iso)

        cov = self._build_covariance(self.prior_context_length + self.prediction_length)
        context_mask = torch.cat(
            (
                torch.ones(self.prior_context_length, dtype=torch.bool),
                torch.zeros(self.prediction_length, dtype=torch.bool),
            )
        )
        future_mask = ~context_mask
        k_context = cov[context_mask][:, context_mask]
        k_cross = cov[context_mask][:, future_mask]
        k_future = cov[future_mask][:, future_mask]
        k_context = k_context + 1e-4 * torch.eye(k_context.shape[0])
        k_inv_cross = torch.linalg.solve(k_context, k_cross)
        cov_reg = k_future - k_cross.transpose(0, 1) @ k_inv_cross
        cov_reg = cov_reg + 1e-5 * torch.eye(cov_reg.shape[0])

        self.register_buffer("k_inv_cross", k_inv_cross.float(), persistent=True)
        self.register_buffer("cov_reg", cov_reg.float(), persistent=True)
        self.register_buffer("chol_reg", torch.linalg.cholesky(cov_reg).float(), persistent=True)

    def _build_covariance(self, length: int) -> Any:
        t = torch.arange(int(length), dtype=torch.float64) * (math.pi / max(self.freq, 1))
        gamma = torch.tensor(self.gamma, dtype=torch.float64)
        if self.kernel in {"iso", "isotropic"}:
            cov = gamma * torch.eye(int(length), dtype=torch.float64)
        elif self.kernel in {"se", "rbf"}:
            cov = torch.exp(-gamma * (t[:, None] - t[None, :]).square())
        elif self.kernel == "ou":
            cov = torch.exp(-gamma * torch.abs(t[:, None] - t[None, :]))
        elif self.kernel in {"pe", "periodic"}:
            cov = torch.exp(-gamma * torch.sin(t[:, None] - t[None, :]).square())
        else:
            raise ValueError("TSFlow prior kernel must be one of 'iso', 'ou', 'se', or 'pe'.")
        return cov + self.iso * torch.eye(int(length), dtype=torch.float64)

    def regression(self, context: Any) -> tuple[Any, Any, Any]:
        """Return GP posterior mean, standard deviation, and one sample.

        ``context`` has shape ``[batch, prior_context_length]``.
        """

        if context.shape[1] != self.prior_context_length:
            raise ValueError(
                f"Expected GP prior context length {self.prior_context_length}, got {context.shape[1]}."
            )
        centered_loc = context.mean(dim=1, keepdim=True)
        if self.kernel in {"pe", "periodic"}:
            centered_loc = torch.zeros_like(centered_loc)
        centered_context = context - centered_loc
        mean = centered_context @ self.k_inv_cross.to(device=context.device, dtype=context.dtype)
        mean = mean + centered_loc.repeat(1, self.prediction_length)
        eps = torch.randn(context.shape[0], self.prediction_length, device=context.device, dtype=context.dtype)
        chol = self.chol_reg.to(device=context.device, dtype=context.dtype)
        sample = mean + eps @ chol.transpose(0, 1)
        std = torch.diagonal(self.cov_reg.to(device=context.device, dtype=context.dtype)).sqrt()
        std = std.unsqueeze(0).expand_as(mean)
        return mean, std, sample


class TSFlowResidualBlock(nn.Module if nn is not None else object):
    """Residual temporal and cross-asset sequence block for the TSFlow backbone."""

    def __init__(
        self,
        hidden_dim: int,
        num_features: int,
        target_dim: int,
        nheads: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        require_torch()
        super().__init__()
        del bidirectional  # The native block uses non-causal full-window attention.
        self.hidden_dim = int(hidden_dim)
        self.target_dim = int(target_dim)
        self.time_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.feature_encoder = nn.Conv2d(int(num_features), self.hidden_dim, kernel_size=1)
        self.temporal_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(nheads),
                dim_feedforward=max(64, 4 * self.hidden_dim),
                dropout=float(dropout),
                activation="gelu",
            ),
            num_layers=1,
        )
        self.asset_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(nheads),
                dim_feedforward=max(64, 4 * self.hidden_dim),
                dropout=float(dropout),
                activation="gelu",
            ),
            num_layers=1,
        )
        self.out_linear1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.out_linear2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)

    def _temporal(self, y: Any) -> Any:
        batch_size, channels, num_assets, length = y.shape
        if length == 1:
            return y
        z = y.permute(3, 0, 2, 1).reshape(length, batch_size * num_assets, channels)
        z = self.temporal_layer(z)
        return z.reshape(length, batch_size, num_assets, channels).permute(1, 3, 2, 0)

    def _assets(self, y: Any) -> Any:
        batch_size, channels, num_assets, length = y.shape
        if num_assets == 1:
            return y
        z = y.permute(2, 0, 3, 1).reshape(num_assets, batch_size * length, channels)
        z = self.asset_layer(z)
        return z.reshape(num_assets, batch_size, length, channels).permute(1, 3, 0, 2)

    def forward(self, x: Any, t: Any, features: Optional[Any]) -> tuple[Any, Any]:
        t = self.time_linear(t).reshape(t.shape[0], self.hidden_dim, 1, 1)
        out = x + t
        out = self._temporal(out)
        out = self._assets(out)
        if features is not None:
            out = out + self.feature_encoder(features)
        out = torch.tanh(out) * torch.sigmoid(out)
        out1 = self.out_linear1(out)
        out2 = self.out_linear2(out)
        return out1 + x, out2


class TSFlowBackbone(nn.Module if nn is not None else object):
    """TSFlow vector-field backbone with flow-time, lag, and asset features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        step_emb: int,
        num_residual_blocks: int,
        num_features: int,
        target_dim: int,
        dropout: float,
        nheads: int,
        init_skip: bool,
        feature_skip: bool,
        bidirectional: bool,
        asset_embedding_dim: int,
    ) -> None:
        require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.target_dim = int(target_dim)
        self.init_skip = bool(init_skip)
        self.feature_skip = bool(feature_skip)
        self.asset_embedding_dim = int(asset_embedding_dim) if self.target_dim > 1 else 0
        self.input_init = nn.Sequential(nn.Linear(int(input_dim), self.hidden_dim), nn.ReLU())
        self.time_init = nn.Sequential(
            nn.Linear(int(step_emb), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        feature_channels = int(num_features) + int(self.feature_skip) + self.asset_embedding_dim
        self.residual_blocks = nn.ModuleList(
            [
                TSFlowResidualBlock(
                    hidden_dim=self.hidden_dim,
                    num_features=feature_channels,
                    target_dim=self.target_dim,
                    nheads=int(nheads),
                    dropout=float(dropout),
                    bidirectional=bool(bidirectional),
                )
                for _ in range(int(num_residual_blocks))
            ]
        )
        self.step_embedding = SinusoidalPositionEmbeddings(int(step_emb))
        self.asset_embedding = nn.Embedding(self.target_dim, self.asset_embedding_dim) if self.asset_embedding_dim else None
        self.out_linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, int(output_dim)),
        )

    def _prepare_features(self, x_in: Any, features: Optional[Any]) -> Optional[Any]:
        if features is None and not self.feature_skip and self.asset_embedding is None:
            return None
        feature_parts = []
        if features is not None:
            feature_parts.append(features.permute(0, 3, 2, 1))
        if self.feature_skip:
            feature_parts.append(x_in.unsqueeze(-1).permute(0, 3, 2, 1))
        if self.asset_embedding is not None:
            batch_size, length, num_assets = x_in.shape
            asset_ids = torch.arange(num_assets, device=x_in.device)
            asset_emb = self.asset_embedding(asset_ids).transpose(0, 1)
            asset_emb = asset_emb.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, length)
            feature_parts.append(asset_emb)
        return torch.cat(feature_parts, dim=1)

    def forward(self, t: Any, x_in: Any, features: Optional[Any] = None) -> Any:
        batch_size, length, num_assets = x_in.shape
        if t.ndim == 0:
            t = t.reshape(1).expand(batch_size)
        else:
            t = t.reshape(t.shape[0], -1)[:, 0]
        t_emb = self.time_init(self.step_embedding(t * 10000.0))
        feature_tensor = self._prepare_features(x_in, features)
        x = self.input_init(x_in.unsqueeze(-1)).permute(0, 3, 2, 1)
        skips = []
        for layer in self.residual_blocks:
            x, skip = layer(x, t_emb, feature_tensor)
            skips.append(skip)
        skip_sum = torch.stack(skips, dim=0).sum(dim=0)
        out = skip_sum.permute(0, 3, 2, 1)
        out = self.out_linear(out)[..., 0]
        if self.init_skip:
            out = out - x_in
        return out


class TSFlowNetwork(nn.Module if nn is not None else object):
    """Conditional TSFlow network over FinProbTS rolling windows."""

    def __init__(
        self,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        setting: str,
        freq: str,
        normalization: Optional[str],
        use_lags: bool,
        lags_seq: Sequence[int],
        use_ema: bool,
        ema_beta: float,
        ema_update_after_step: int,
        ema_update_every: int,
        num_steps: int,
        solver: str,
        matching: str,
        sigm: float,
        guidance_scale: float,
        prior_kernel: str,
        prior_gamma: float,
        prior_context_freqs: int,
        prior_iso: Optional[float],
        backbone_params: Dict[str, Any],
    ) -> None:
        require_torch()
        super().__init__()
        if str(setting).lower() != "multivariate":
            raise NotImplementedError("FinProbTS TSFlow currently supports setting='multivariate'.")
        if str(solver).lower() != "euler":
            raise NotImplementedError("FinProbTS TSFlow currently implements Euler sampling.")
        if str(matching).lower() != "random":
            raise NotImplementedError("FinProbTS TSFlow currently implements random conditional flow matching.")
        self.target_dim = int(target_dim)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.sequence_length = self.context_length + self.prediction_length
        self.freq = str(freq)
        self.normalization = None if normalization is None else str(normalization).lower()
        self.use_lags = bool(use_lags)
        self.lags_seq = [int(lag) for lag in lags_seq]
        self.use_ema = bool(use_ema)
        self.ema_beta = float(ema_beta)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_update_every = max(1, int(ema_update_every))
        self.num_steps = int(num_steps)
        self.solver = str(solver)
        self.matching = str(matching)
        self.sigmin = float(sigm)
        self.guidance_scale = float(guidance_scale)
        self.season_length = _season_length(freq)
        self.prior_context_length = max(1, int(prior_context_freqs) * self.prediction_length)
        if self.prior_context_length > self.context_length:
            self.prior_context_length = self.context_length
        self.sigmax = self.sigmin if str(prior_kernel).lower() in {"iso", "isotropic"} else 1.0
        iso = (1e-2 if str(prior_kernel).lower() in {"iso", "isotropic"} else 1e-1) if prior_iso is None else float(prior_iso)
        self.q0 = GaussianProcessPrior(
            kernel=str(prior_kernel),
            context_freqs=max(1, self.prior_context_length // max(self.prediction_length, 1)),
            prediction_length=self.prediction_length,
            freq=self.season_length,
            gamma=float(prior_gamma),
            iso=iso,
        )
        self.prior_context_length = self.q0.prior_context_length

        num_features = 2 + (len(self.lags_seq) if self.use_lags else 0)
        self.backbone = TSFlowBackbone(
            input_dim=int(backbone_params.get("input_dim", 1)),
            hidden_dim=int(backbone_params.get("hidden_dim", 64)),
            output_dim=int(backbone_params.get("output_dim", 1)),
            step_emb=int(backbone_params.get("step_emb", 64)),
            num_residual_blocks=int(backbone_params.get("num_residual_blocks", 3)),
            num_features=num_features,
            target_dim=self.target_dim,
            dropout=float(backbone_params.get("dropout", 0.0)),
            nheads=int(backbone_params.get("nheads", 8)),
            init_skip=bool(backbone_params.get("init_skip", False)),
            feature_skip=bool(backbone_params.get("feature_skip", True)),
            bidirectional=bool(backbone_params.get("bidirectional", True)),
            asset_embedding_dim=int(backbone_params.get("asset_embedding_dim", 16)),
        )
        self.ema_backbone = copy.deepcopy(self.backbone) if self.use_ema else None
        if self.ema_backbone is not None:
            for param in self.ema_backbone.parameters():
                param.requires_grad_(False)
        self._ema_steps = 0

    def _scale(self, past: Any, past_observed: Any) -> tuple[Any, Any]:
        if self.normalization in {"longmean", "mean"}:
            denom = past_observed.sum(dim=1).clamp_min(1.0)
            scale = (past.abs() * past_observed).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
            return torch.zeros_like(scale), scale.clamp_min(1.0)
        if self.normalization == "zscore":
            denom = past_observed.sum(dim=1).clamp_min(1.0)
            loc = (past * past_observed).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
            var = ((past - loc).square() * past_observed).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
            return loc, var.sqrt().clamp_min(1.0)
        return torch.zeros(past.shape[0], 1, past.shape[-1], device=past.device, dtype=past.dtype), torch.ones(
            past.shape[0], 1, past.shape[-1], device=past.device, dtype=past.dtype
        )

    def _lag_features(self, sequence: Any) -> Any:
        batch_size, length, target_dim = sequence.shape
        features = []
        for lag in self.lags_seq:
            lagged = torch.zeros(batch_size, length, target_dim, device=sequence.device, dtype=sequence.dtype)
            if lag < length:
                lagged[:, lag:, :] = sequence[:, :-lag, :]
            features.append(lagged.unsqueeze(-1))
        return torch.cat(features, dim=-1)

    def _extract_features(self, batch: Dict[str, Any], sample_prior: bool) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        past = batch["past_target"]
        future = batch["future_target"]
        past_observed = batch["past_observed_values"]
        future_observed = batch["future_observed_values"]
        loc, scale = self._scale(past, past_observed)
        scaled_past = (past - loc) / scale
        scaled_future = (future - loc) / scale

        prior_context = scaled_past[:, -self.prior_context_length :, :]
        gp_context = prior_context.permute(0, 2, 1).reshape(-1, self.prior_context_length)
        gp_mean, gp_std, gp_sample = self.q0.regression(gp_context)
        gp_mean = gp_mean.reshape(past.shape[0], self.target_dim, self.prediction_length).permute(0, 2, 1)
        gp_std = gp_std.reshape(past.shape[0], self.target_dim, self.prediction_length).permute(0, 2, 1)
        gp_sample = gp_sample.reshape(past.shape[0], self.target_dim, self.prediction_length).permute(0, 2, 1)

        x1 = torch.cat((scaled_past, scaled_future), dim=1)
        x0_future = gp_sample if sample_prior else gp_mean
        x0 = torch.cat((scaled_past, x0_future), dim=1)
        observation_mask = torch.cat((past_observed, torch.zeros_like(future_observed)), dim=1)
        loss_mask = torch.cat((past_observed, future_observed), dim=1)

        feature_parts = []
        if self.use_lags:
            feature_parts.append(self._lag_features(x1))
        feature_parts.append(torch.cat((scaled_past, gp_mean), dim=1).unsqueeze(-1))
        feature_parts.append(observation_mask.unsqueeze(-1))
        features = torch.cat(feature_parts, dim=-1)
        return x1, x0, loss_mask, observation_mask, loc, scale, features

    def forward_path(self, x1: Any, x0: Any, t: Any) -> tuple[Any, Any]:
        t_expanded = t.reshape(t.shape[0], 1, 1)
        eps = torch.randn_like(x0)
        sig_t = (1.0 - t_expanded) * self.sigmax + t_expanded * self.sigmin
        psi = t_expanded * x1 + (1.0 - t_expanded) * x0 + sig_t * eps
        dpsi = x1 - x0 + (self.sigmin - self.sigmax) * eps
        return psi, dpsi

    def loss(self, batch: Dict[str, Any]) -> Any:
        x1, x0, loss_mask, _, _, _, features = self._extract_features(batch, sample_prior=True)
        t = torch.rand(x1.shape[0], 1, device=x1.device, dtype=x1.dtype)
        psi, dpsi = self.forward_path(x1, x0, t)
        predicted = self.backbone(t, psi, features)
        sq_error = (predicted - dpsi).square()
        denom = loss_mask.sum().clamp_min(1.0)
        return (sq_error * loss_mask).sum() / denom

    def update_ema(self) -> None:
        if self.ema_backbone is None:
            return
        self._ema_steps += 1
        if self._ema_steps < self.ema_update_after_step:
            self.ema_backbone.load_state_dict(self.backbone.state_dict())
            return
        if self._ema_steps % self.ema_update_every != 0:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_backbone.parameters(), self.backbone.parameters()):
                ema_param.mul_(self.ema_beta).add_(param.detach(), alpha=1.0 - self.ema_beta)

    def sample(self, batch: Dict[str, Any], num_samples: int) -> Any:
        past = batch["past_target"].repeat_interleave(int(num_samples), dim=0)
        future = torch.zeros(
            past.shape[0],
            self.prediction_length,
            self.target_dim,
            device=past.device,
            dtype=past.dtype,
        )
        past_observed = batch["past_observed_values"].repeat_interleave(int(num_samples), dim=0)
        future_observed = torch.zeros_like(future)
        repeated = {
            "past_target": past,
            "future_target": future,
            "past_observed_values": past_observed,
            "future_observed_values": future_observed,
        }
        _, x0, _, _, loc, scale, features = self._extract_features(repeated, sample_prior=True)
        x = x0 + self.sigmax * torch.randn_like(x0)
        backbone = self.ema_backbone if self.ema_backbone is not None and self.use_ema else self.backbone
        dt = 1.0 / float(max(self.num_steps, 1))
        for step in range(max(self.num_steps, 1)):
            t_value = (float(step) + 0.5) * dt
            t = torch.full((x.shape[0], 1), t_value, device=x.device, dtype=x.dtype)
            x = x + dt * backbone(t, x, features)
        x = x * scale + loc
        future = x[:, self.context_length :, :]
        batch_size = batch["past_target"].shape[0]
        return future.reshape(batch_size, int(num_samples), self.prediction_length, self.target_dim)


class TSFlowForecastModel(BaseProbForecastModel):
    """Native TSFlow conditional flow-matching forecaster."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        freq: str = "B",
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        context_length: Optional[int] = None,
        setting: str = "multivariate",
        normalization: Optional[str] = "longmean",
        use_lags: bool = True,
        lags_seq: Optional[Sequence[int]] = None,
        use_ema: bool = True,
        ema_beta: float = 0.9999,
        ema_update_after_step: int = 128,
        ema_update_every: int = 1,
        num_steps: int = 32,
        solver: str = "euler",
        matching: str = "random",
        sigm: float = 1e-3,
        sigmin: Optional[float] = None,
        sigmax: Optional[float] = None,
        guidance_scale: float = 0.0,
        prior_kernel: str = "ou",
        prior_gamma: float = 1.0,
        prior_context_freqs: int = 14,
        prior_iso: Optional[float] = None,
        hidden_dim: int = 64,
        step_emb: int = 64,
        num_residual_blocks: int = 3,
        residual_block: str = "s4",
        dropout: float = 0.0,
        init_skip: bool = False,
        feature_skip: bool = True,
        bidirectional: bool = True,
        nheads: int = 8,
        asset_embedding_dim: int = 16,
        batch_size: int = 32,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
        gradient_clip_val: Optional[float] = 0.5,
        patience: Optional[int] = 10,
        device: str = "auto",
        seed: Optional[int] = None,
        scaling: bool = False,
        scaler_min_std: float = 1e-6,
        verbose: bool = False,
        backbone_params: Optional[Dict[str, Any]] = None,
        prior_params: Optional[Dict[str, Any]] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        ema_params: Optional[Dict[str, Any]] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        rnn_type: Optional[str] = None,
        time_embedding_dim: Optional[int] = None,
        vector_hidden_size: Optional[int] = None,
        num_vector_layers: Optional[int] = None,
        flow_steps: Optional[int] = None,
        prior_scale: Optional[float] = None,
        prior_min_scale: Optional[float] = None,
        **_: Any,
    ) -> None:
        del rnn_type, prior_scale, prior_min_scale
        if sigmin is not None:
            sigm = float(sigmin)
        if sigmax is not None and abs(float(sigmax) - 1.0) > 1e-12:
            raise NotImplementedError("FinProbTS TSFlow derives sigmax from the prior, following upstream TSFlow.")
        if hidden_size is not None:
            hidden_dim = int(hidden_size)
        if vector_hidden_size is not None:
            hidden_dim = int(vector_hidden_size)
        if num_layers is not None:
            num_residual_blocks = int(num_layers)
        if num_vector_layers is not None:
            num_residual_blocks = int(num_vector_layers)
        if time_embedding_dim is not None:
            step_emb = int(time_embedding_dim)
        if flow_steps is not None:
            num_steps = int(flow_steps)
        if lr is not None:
            learning_rate = float(lr)
        if optimizer_params:
            learning_rate = float(optimizer_params.get("lr", learning_rate))
            weight_decay = float(optimizer_params.get("weight_decay", weight_decay))
            patience = optimizer_params.get("patience", patience)
        if ema_params:
            ema_beta = float(ema_params.get("beta", ema_beta))
            ema_update_after_step = int(ema_params.get("update_after_step", ema_update_after_step))
            ema_update_every = int(ema_params.get("update_every", ema_update_every))
        if prior_params:
            prior_kernel = str(prior_params.get("kernel", prior_kernel))
            prior_gamma = float(prior_params.get("gamma", prior_gamma))
            prior_context_freqs = int(prior_params.get("context_freqs", prior_context_freqs))
            prior_iso = prior_params.get("iso", prior_iso)
        if backbone_params:
            hidden_dim = int(backbone_params.get("hidden_dim", hidden_dim))
            step_emb = int(backbone_params.get("step_emb", step_emb))
            num_residual_blocks = int(backbone_params.get("num_residual_blocks", num_residual_blocks))
            residual_block = str(backbone_params.get("residual_block", residual_block))
            dropout = float(backbone_params.get("dropout", dropout))
            init_skip = bool(backbone_params.get("init_skip", init_skip))
            feature_skip = bool(backbone_params.get("feature_skip", feature_skip))
            bidirectional = bool(backbone_params.get("bidirectional", bidirectional))
            nheads = int(backbone_params.get("nheads", nheads))
            asset_embedding_dim = int(backbone_params.get("asset_embedding_dim", asset_embedding_dim))

        if str(residual_block).lower() != "s4":
            raise ValueError("TSFlow residual_block must be 's4'; FinProbTS approximates it with native sequence blocks.")
        if int(hidden_dim) % int(nheads) != 0:
            raise ValueError("hidden_dim must be divisible by nheads.")

        self.input_size = None if input_size is None else int(input_size)
        self.freq = str(freq)
        self.prediction_length = None if prediction_length is None else int(prediction_length)
        self.target_dim = None if target_dim is None else int(target_dim)
        self.context_length = None if context_length is None else int(context_length)
        self.setting = str(setting)
        self.normalization = normalization
        self.use_lags = bool(use_lags)
        self._configured_lags = None if lags_seq is None else [int(lag) for lag in lags_seq]
        self.use_ema = bool(use_ema)
        self.ema_beta = float(ema_beta)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_update_every = int(ema_update_every)
        self.num_steps = int(num_steps)
        self.solver = str(solver)
        self.matching = str(matching)
        self.sigm = float(sigm)
        self.guidance_scale = float(guidance_scale)
        self.prior_kernel = str(prior_kernel)
        self.prior_gamma = float(prior_gamma)
        self.prior_context_freqs = int(prior_context_freqs)
        self.prior_iso = None if prior_iso is None else float(prior_iso)
        self.hidden_dim = int(hidden_dim)
        self.step_emb = int(step_emb)
        self.num_residual_blocks = int(num_residual_blocks)
        self.residual_block = str(residual_block)
        self.dropout = float(dropout)
        self.init_skip = bool(init_skip)
        self.feature_skip = bool(feature_skip)
        self.bidirectional = bool(bidirectional)
        self.nheads = int(nheads)
        self.asset_embedding_dim = int(asset_embedding_dim)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.lr = self.learning_rate
        self.weight_decay = float(weight_decay)
        self.gradient_clip_val = None if gradient_clip_val is None else float(gradient_clip_val)
        self.patience = None if patience is None else int(patience)
        self.device_name = device
        self.seed = seed
        self.scaling = bool(scaling)
        self.scaler_min_std = float(scaler_min_std)
        self.verbose = bool(verbose)

        self._network: Optional[TSFlowNetwork] = None
        self._device = None
        self._scaler: Optional[TorchStandardScaler] = None
        self._num_assets: Optional[int] = None
        self._fit_context_length: Optional[int] = None
        self._fit_prediction_length: Optional[int] = None
        self._asset_ids: Optional[list[str]] = None
        self._lags_seq: Optional[list[int]] = None
        self._is_fitted = False
        self.training_history: list[Dict[str, float]] = []

    def _init_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "freq": self.freq,
            "prediction_length": self.prediction_length,
            "target_dim": self.target_dim,
            "context_length": self.context_length,
            "setting": self.setting,
            "normalization": self.normalization,
            "use_lags": self.use_lags,
            "lags_seq": None if self._configured_lags is None else list(self._configured_lags),
            "use_ema": self.use_ema,
            "ema_beta": self.ema_beta,
            "ema_update_after_step": self.ema_update_after_step,
            "ema_update_every": self.ema_update_every,
            "num_steps": self.num_steps,
            "solver": self.solver,
            "matching": self.matching,
            "sigm": self.sigm,
            "guidance_scale": self.guidance_scale,
            "prior_kernel": self.prior_kernel,
            "prior_gamma": self.prior_gamma,
            "prior_context_freqs": self.prior_context_freqs,
            "prior_iso": self.prior_iso,
            "hidden_dim": self.hidden_dim,
            "step_emb": self.step_emb,
            "num_residual_blocks": self.num_residual_blocks,
            "residual_block": self.residual_block,
            "dropout": self.dropout,
            "init_skip": self.init_skip,
            "feature_skip": self.feature_skip,
            "bidirectional": self.bidirectional,
            "nheads": self.nheads,
            "asset_embedding_dim": self.asset_embedding_dim,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_val": self.gradient_clip_val,
            "patience": self.patience,
            "device": self.device_name,
            "seed": self.seed,
            "scaling": self.scaling,
            "scaler_min_std": self.scaler_min_std,
            "verbose": self.verbose,
        }

    def _validate_data(self, data: RollingWindowDataset) -> None:
        if self.context_length is not None and data.context_length != self.context_length:
            raise ValueError(
                f"TSFlow context_length={self.context_length} does not match data context_length={data.context_length}."
            )
        if self.prediction_length is not None and data.prediction_length != self.prediction_length:
            raise ValueError(
                f"TSFlow prediction_length={self.prediction_length} does not match data prediction_length={data.prediction_length}."
            )
        if self.target_dim is not None and data.num_assets != self.target_dim:
            raise ValueError(f"TSFlow target_dim={self.target_dim} does not match data num_assets={data.num_assets}.")

    def _build_network(self, num_assets: int, context_length: int, prediction_length: int) -> None:
        self._num_assets = int(num_assets)
        self._fit_context_length = int(context_length)
        self._fit_prediction_length = int(prediction_length)
        self._lags_seq = _normalize_lags(self._configured_lags, self.freq, context_length) if self.use_lags else []
        effective_input_size = 1
        if self.input_size is not None and self.input_size != effective_input_size:
            raise ValueError(f"Configured input_size={self.input_size} does not match TSFlow input_dim=1.")
        self._device = resolve_torch_device(self.device_name)
        self._network = TSFlowNetwork(
            target_dim=num_assets,
            context_length=context_length,
            prediction_length=prediction_length,
            setting=self.setting,
            freq=self.freq,
            normalization=self.normalization,
            use_lags=self.use_lags,
            lags_seq=self._lags_seq,
            use_ema=self.use_ema,
            ema_beta=self.ema_beta,
            ema_update_after_step=self.ema_update_after_step,
            ema_update_every=self.ema_update_every,
            num_steps=self.num_steps,
            solver=self.solver,
            matching=self.matching,
            sigm=self.sigm,
            guidance_scale=self.guidance_scale,
            prior_kernel=self.prior_kernel,
            prior_gamma=self.prior_gamma,
            prior_context_freqs=self.prior_context_freqs,
            prior_iso=self.prior_iso,
            backbone_params={
                "input_dim": 1,
                "hidden_dim": self.hidden_dim,
                "output_dim": 1,
                "step_emb": self.step_emb,
                "num_residual_blocks": self.num_residual_blocks,
                "residual_block": self.residual_block,
                "dropout": self.dropout,
                "init_skip": self.init_skip,
                "feature_skip": self.feature_skip,
                "bidirectional": self.bidirectional,
                "nheads": self.nheads,
                "asset_embedding_dim": self.asset_embedding_dim,
            },
        ).to(self._device)

    def _make_loader(self, data: RollingWindowDataset, shuffle: bool) -> Any:
        return make_torch_data_loader(data, self.batch_size, shuffle, self._scaler, include_time_features=False)

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
        self._scaler = TorchStandardScaler.fit(train_data.x_context, min_std=self.scaler_min_std) if self.scaling else None
        self._build_network(train_data.num_assets, train_data.context_length, train_data.prediction_length)
        train_loader = self._make_loader(train_data, shuffle=True)
        val_loader = self._make_loader(val_data, shuffle=False) if val_data is not None and len(val_data) > 0 else None
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

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
                self._network.update_ema()
                total += float(loss.detach().cpu())
                count += 1
            train_loss = total / max(count, 1)
            val_loss = self._evaluate(val_loader) if val_loader is not None else train_loss
            self.training_history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            if self.verbose:
                print(f"TSFlow epoch {epoch}/{self.max_epochs}: train={train_loss:.6f} val={val_loss:.6f}")
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
            raise ValueError("test_data context_length does not match the fitted TSFlow model.")
        if self._fit_prediction_length != test_data.prediction_length:
            raise ValueError("test_data prediction_length does not match the fitted TSFlow model.")

        loader = self._make_loader(test_data, shuffle=False)
        chunks = []
        self._network.eval()
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                chunks.append(self._network.sample(batch, int(num_samples)).cpu().numpy())
        samples = np.concatenate(chunks, axis=0)
        if self._scaler is not None:
            samples = self._scaler.inverse_transform_array(samples)
        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "tsflow",
                "implementation": "finprobts_native_tsflow",
                "reference": "marcelkollovieh/TSFlow",
                "seed": self.seed,
                "setting": self.setting,
                "freq": self.freq,
                "normalization": self.normalization,
                "use_lags": self.use_lags,
                "lags_seq": list(self._lags_seq or []),
                "use_ema": self.use_ema,
                "num_steps": self.num_steps,
                "solver": self.solver,
                "matching": self.matching,
                "sigmin": self.sigm,
                "prior_kernel": self.prior_kernel,
                "prior_gamma": self.prior_gamma,
                "prior_context_freqs": self.prior_context_freqs,
                "hidden_dim": self.hidden_dim,
                "num_residual_blocks": self.num_residual_blocks,
                "residual_block": self.residual_block,
                "model_internal_scaling": self.scaling,
                "training_history": list(self.training_history),
            },
        )

    def save(self, path: str) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("Cannot save an unfitted TSFlowForecastModel.")
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
                "lags_seq": self._lags_seq,
                "scaler_state": self._scaler.state_dict() if self._scaler is not None else None,
                "training_history": self.training_history,
                "is_fitted": self._is_fitted,
            },
            output_dir / "model.pt",
        )

    @classmethod
    def load(cls, path: str) -> "TSFlowForecastModel":
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
        model._lags_seq = [int(lag) for lag in payload.get("lags_seq", model._lags_seq or [])]
        scaler_state = payload.get("scaler_state")
        model._scaler = TorchStandardScaler.from_state_dict(scaler_state) if scaler_state is not None else None
        model.training_history = list(payload.get("training_history", []))
        model._is_fitted = bool(payload.get("is_fitted", True))
        return model
