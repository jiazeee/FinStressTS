"""DeepVAR forecaster for the FinProbTS rolling-window contract.

The implementation follows the main GluonTS DeepVAR design: lagged
multivariate target inputs, recurrent state dynamics, optional time/static
features, local target scaling, a low-rank multivariate Gaussian output, and
autoregressive sampling over the forecast horizon. The public API remains the
FinProbTS ``BaseProbForecastModel`` interface.
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


try:  # Keep torch optional for the core package.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None
    F = None


def _default_lags_for_frequency(freq: Optional[str]) -> list[int]:
    """Return a compact GluonTS-style lag set for common frequencies."""

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


class DeepVARNetwork(nn.Module if nn is not None else object):
    """Autoregressive RNN with a low-rank multivariate Gaussian head."""

    def __init__(
        self,
        num_assets: int,
        time_feature_dim: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str,
        dropout: float,
        rank: int,
        lags_seq: Sequence[int],
        embedding_dim: int,
        use_scale_feature: bool,
        scaling: bool,
        minimum_scale: float,
        min_distribution_scale: float,
        jitter: float,
    ) -> None:
        require_torch()
        super().__init__()
        self.num_assets = int(num_assets)
        self.time_feature_dim = int(time_feature_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.rank = max(0, int(rank))
        self.lags_seq = [int(lag) for lag in lags_seq]
        self.max_lag = max(self.lags_seq)
        self.embedding_dim = max(0, int(embedding_dim))
        self.use_scale_feature = bool(use_scale_feature)
        self.scaling = bool(scaling)
        self.minimum_scale = float(minimum_scale)
        self.min_distribution_scale = float(min_distribution_scale)
        self.jitter = float(jitter)

        lagged_target_dim = self.num_assets * len(self.lags_seq)
        static_dim = self.num_assets * self.embedding_dim
        scale_dim = self.num_assets if self.use_scale_feature else 0
        input_size = lagged_target_dim + self.time_feature_dim + static_dim + scale_dim

        self.asset_embedding = (
            nn.Embedding(self.num_assets, self.embedding_dim)
            if self.embedding_dim > 0
            else None
        )
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}.get(str(rnn_type).upper())
        if rnn_cls is None:
            raise ValueError("rnn_type must be 'GRU' or 'LSTM'.")
        self.rnn_type = str(rnn_type).upper()
        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.loc_proj = nn.Linear(self.hidden_size, self.num_assets)
        self.scale_proj = nn.Linear(self.hidden_size, self.num_assets)
        self.factor_proj = (
            nn.Linear(self.hidden_size, self.num_assets * self.rank)
            if self.rank > 0
            else None
        )

    def _initial_state(self, batch_size: int, device: Any) -> Any:
        shape = (self.num_layers, int(batch_size), self.hidden_size)
        h = torch.zeros(shape, device=device)
        if self.rnn_type == "LSTM":
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
            mean_abs.clamp_min(self.minimum_scale),
            torch.ones_like(mean_abs),
        )
        return scale.unsqueeze(1)

    def _static_features(self, batch_size: int, scale: Any, device: Any) -> list[Any]:
        features = []
        if self.asset_embedding is not None:
            indices = torch.arange(self.num_assets, device=device)
            embedded = self.asset_embedding(indices).reshape(1, -1)
            features.append(embedded.expand(batch_size, -1))
        if self.use_scale_feature:
            features.append(scale.squeeze(1).clamp_min(self.minimum_scale).log())
        return features

    def _features_from_indices(
        self,
        history: Any,
        step_index: int,
        time_feat: Any,
        scale: Any,
    ) -> Any:
        lagged = [history[:, step_index - lag, :] for lag in self.lags_seq]
        parts = [torch.cat(lagged, dim=-1)]
        if self.time_feature_dim:
            parts.append(time_feat)
        parts.extend(self._static_features(history.shape[0], scale, history.device))
        return torch.cat(parts, dim=-1)

    def _features_from_tail(self, history: Any, time_feat: Any, scale: Any) -> Any:
        lagged = [history[:, -lag, :] for lag in self.lags_seq]
        parts = [torch.cat(lagged, dim=-1)]
        if self.time_feature_dim:
            parts.append(time_feat)
        parts.extend(self._static_features(history.shape[0], scale, history.device))
        return torch.cat(parts, dim=-1)

    def _encode_context(self, past_scaled: Any, past_time_feat: Any, scale: Any) -> Any:
        batch_size, context_length, _ = past_scaled.shape
        if context_length < self.max_lag:
            raise ValueError(
                f"context_length={context_length} must be at least max(lags_seq)={self.max_lag}."
            )
        if context_length == self.max_lag:
            return self._initial_state(batch_size, past_scaled.device)

        inputs = [
            self._features_from_indices(
                past_scaled,
                step_index=t,
                time_feat=past_time_feat[:, t, :],
                scale=scale,
            )
            for t in range(self.max_lag, context_length)
        ]
        rnn_input = torch.stack(inputs, dim=1)
        _, state = self.rnn(rnn_input, self._initial_state(batch_size, past_scaled.device))
        return state

    def _distribution_from_state(self, rnn_output: Any) -> Any:
        loc = self.loc_proj(rnn_output)
        diag_scale = F.softplus(self.scale_proj(rnn_output)) + self.min_distribution_scale
        cov_diag = diag_scale.square() + self.jitter
        if self.factor_proj is not None:
            cov_factor = self.factor_proj(rnn_output).reshape(-1, self.num_assets, self.rank)
            return torch.distributions.LowRankMultivariateNormal(
                loc=loc,
                cov_factor=cov_factor,
                cov_diag=cov_diag,
            )
        scale_tril = torch.diag_embed(cov_diag.sqrt())
        return torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)

    def _next_distribution(self, history: Any, time_feat: Any, scale: Any, state: Any) -> tuple[Any, Any]:
        step_input = self._features_from_tail(history, time_feat, scale).unsqueeze(1)
        output, next_state = self.rnn(step_input, state)
        return self._distribution_from_state(output[:, 0, :]), next_state

    def negative_log_likelihood(self, batch: Dict[str, Any]) -> Any:
        past = batch["past_target"]
        future = batch["future_target"]
        past_observed = batch["past_observed_values"]
        future_observed = batch["future_observed_values"]
        scale = self._compute_scale(past, past_observed)
        past_scaled = past / scale
        future_scaled = future / scale

        state = self._encode_context(past_scaled, batch["past_time_feat"], scale)
        history = past_scaled
        weighted_losses = []
        weights = []
        for step in range(future_scaled.shape[1]):
            dist, state = self._next_distribution(
                history=history,
                time_feat=batch["future_time_feat"][:, step, :],
                scale=scale,
                state=state,
            )
            target = future_scaled[:, step, :]
            observed_weight = future_observed[:, step, :].min(dim=-1).values
            nll = -dist.log_prob(target)
            weighted_losses.append(nll * observed_weight)
            weights.append(observed_weight)
            history = torch.cat((history, target.unsqueeze(1)), dim=1)

        loss_sum = torch.stack(weighted_losses, dim=0).sum()
        weight_sum = torch.stack(weights, dim=0).sum()
        if float(weight_sum.detach().cpu()) <= 0.0:
            return loss_sum * 0.0
        return loss_sum / weight_sum

    def sample(self, batch: Dict[str, Any], num_samples: int) -> Any:
        past = batch["past_target"]
        past_observed = batch["past_observed_values"]
        future_time_feat = batch["future_time_feat"]
        prediction_length = future_time_feat.shape[1]
        batch_size = past.shape[0]

        scale = self._compute_scale(past, past_observed)
        past_scaled = past / scale
        state = self._encode_context(past_scaled, batch["past_time_feat"], scale)

        repeats = int(num_samples)
        history = past_scaled.repeat_interleave(repeats, dim=0)
        scale_repeated = scale.repeat_interleave(repeats, dim=0)
        state = self._repeat_state(state, repeats)

        sample_steps = []
        for step in range(prediction_length):
            time_feat = future_time_feat[:, step, :].repeat_interleave(repeats, dim=0)
            dist, state = self._next_distribution(
                history=history,
                time_feat=time_feat,
                scale=scale_repeated,
                state=state,
            )
            sample = dist.sample()
            sample_steps.append(sample)
            history = torch.cat((history, sample.unsqueeze(1)), dim=1)

        scaled_samples = torch.stack(sample_steps, dim=1)
        samples = scaled_samples.reshape(batch_size, repeats, prediction_length, self.num_assets)
        return samples * scale.unsqueeze(1)


class DeepVARForecastModel(BaseProbForecastModel):
    """Faithful DeepVAR-style model behind the FinProbTS model interface."""

    def __init__(
        self,
        hidden_size: int = 40,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        rank: int = 5,
        embedding_dim: int = 5,
        context_length: Optional[int] = None,
        num_parallel_samples: int = 100,
        cardinality: Optional[Sequence[int]] = None,
        conditioning_length: int = 200,
        use_marginal_transformation: bool = False,
        pick_incomplete: bool = False,
        distr_output: Optional[Any] = None,
        time_features: Optional[Any] = None,
        train_sampler: Optional[Any] = None,
        validation_sampler: Optional[Any] = None,
        lags_seq: Optional[Sequence[int]] = None,
        freq: Optional[str] = None,
        use_time_features: bool = True,
        use_scale_feature: bool = True,
        batch_size: int = 32,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-8,
        gradient_clip_val: Optional[float] = 10.0,
        patience: Optional[int] = 10,
        device: str = "auto",
        seed: Optional[int] = None,
        scaling: bool = True,
        min_scale: float = 1e-5,
        scaler_min_std: float = 1e-6,
        jitter: float = 1e-6,
        verbose: bool = False,
        num_cells: Optional[int] = None,
        cell_type: Optional[str] = None,
        dropout_rate: Optional[float] = None,
        embedding_dimension: Optional[int] = None,
        lr: Optional[float] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if num_cells is not None:
            hidden_size = int(num_cells)
        if cell_type is not None:
            rnn_type = str(cell_type)
        if dropout_rate is not None:
            dropout = float(dropout_rate)
        if embedding_dimension is not None:
            embedding_dim = int(embedding_dimension)
        if lr is not None:
            learning_rate = float(lr)
        if optim_kwargs:
            learning_rate = float(optim_kwargs.get("lr", learning_rate))
            weight_decay = float(optim_kwargs.get("weight_decay", weight_decay))
            patience = optim_kwargs.get("patience", patience)
        unsupported = {
            "distr_output": distr_output,
            "time_features": time_features,
            "train_sampler": train_sampler,
            "validation_sampler": validation_sampler,
        }
        unsupported = {name: value for name, value in unsupported.items() if value is not None}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                "This native FinProbTS DeepVAR currently follows the upstream defaults "
                f"for these options but does not support custom {names} yet."
            )
        if bool(use_marginal_transformation):
            raise NotImplementedError(
                "DeepVAR use_marginal_transformation=True is an upstream option, "
                "but the native FinProbTS implementation has not added the "
                "CDF-to-Gaussian transform yet."
            )
        if bool(pick_incomplete):
            raise NotImplementedError(
                "DeepVAR pick_incomplete=True is an upstream sampler option, "
                "but FinProbTS currently builds fixed complete rolling windows."
            )
        if cardinality is not None and [int(item) for item in cardinality] != [1]:
            raise NotImplementedError(
                "Custom DeepVAR static categorical cardinality is not supported yet; "
                "the upstream default cardinality=[1] is supported."
            )

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.rnn_type = str(rnn_type)
        self.dropout = float(dropout)
        self.rank = int(rank)
        self.embedding_dim = int(embedding_dim)
        self.context_length = None if context_length is None else int(context_length)
        self.num_parallel_samples = int(num_parallel_samples)
        self.cardinality = [1] if cardinality is None else [int(item) for item in cardinality]
        self.conditioning_length = int(conditioning_length)
        self.use_marginal_transformation = bool(use_marginal_transformation)
        self.pick_incomplete = bool(pick_incomplete)
        self.freq = None if freq is None else str(freq)
        self.lags_seq = _normalize_lags(lags_seq, self.freq)
        self.use_time_features = bool(use_time_features)
        self.use_scale_feature = bool(use_scale_feature)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip_val = None if gradient_clip_val is None else float(gradient_clip_val)
        self.patience = None if patience is None else int(patience)
        self.device_name = device
        self.seed = seed
        self.scaling = bool(scaling)
        self.min_scale = float(min_scale)
        self.scaler_min_std = float(scaler_min_std)
        self.jitter = float(jitter)
        self.verbose = bool(verbose)

        self._network: Optional[DeepVARNetwork] = None
        self._device = None
        self._num_assets: Optional[int] = None
        self._asset_ids: Optional[list[str]] = None
        self._prediction_length: Optional[int] = None
        self._time_feature_dim: Optional[int] = None
        self._is_fitted = False
        self.training_history: list[Dict[str, float]] = []

    def _init_params(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "dropout": self.dropout,
            "rank": self.rank,
            "embedding_dim": self.embedding_dim,
            "context_length": self.context_length,
            "num_parallel_samples": self.num_parallel_samples,
            "cardinality": list(self.cardinality),
            "conditioning_length": self.conditioning_length,
            "use_marginal_transformation": self.use_marginal_transformation,
            "pick_incomplete": self.pick_incomplete,
            "freq": self.freq,
            "lags_seq": list(self.lags_seq),
            "use_time_features": self.use_time_features,
            "use_scale_feature": self.use_scale_feature,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_val": self.gradient_clip_val,
            "patience": self.patience,
            "device": self.device_name,
            "seed": self.seed,
            "scaling": self.scaling,
            "min_scale": self.min_scale,
            "scaler_min_std": self.scaler_min_std,
            "jitter": self.jitter,
            "verbose": self.verbose,
        }

    def _validate_data(self, data: RollingWindowDataset) -> None:
        if self.context_length is not None and data.context_length != self.context_length:
            raise ValueError(
                "DeepVAR context_length is controlled by the FinProbTS task window. "
                f"Got model context_length={self.context_length} but data has "
                f"context_length={data.context_length}."
            )
        if data.context_length < max(self.lags_seq):
            raise ValueError(
                f"DeepVAR requires context_length >= max(lags_seq). "
                f"Got context_length={data.context_length}, lags_seq={self.lags_seq}."
            )
        if data.prediction_length <= 0:
            raise ValueError("prediction_length must be positive.")

    def _build_network(self, num_assets: int, time_feature_dim: int) -> None:
        require_torch()
        self._num_assets = int(num_assets)
        self._time_feature_dim = int(time_feature_dim)
        self._device = resolve_torch_device(self.device_name)
        self._network = DeepVARNetwork(
            num_assets=num_assets,
            time_feature_dim=time_feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
            rank=self.rank,
            lags_seq=self.lags_seq,
            embedding_dim=self.embedding_dim,
            use_scale_feature=self.use_scale_feature,
            scaling=self.scaling,
            minimum_scale=self.scaler_min_std,
            min_distribution_scale=self.min_scale,
            jitter=self.jitter,
        ).to(self._device)

    def _make_loader(self, data: RollingWindowDataset, shuffle: bool) -> Any:
        return make_torch_data_loader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            scaler=None,
            include_time_features=self.use_time_features,
        )

    def _run_epoch(self, loader: Any, optimizer: Any) -> float:
        assert self._network is not None and self._device is not None
        self._network.train()
        total_loss = 0.0
        total_batches = 0
        for batch in iter_torch_batches(loader, self._device):
            optimizer.zero_grad()
            loss = self._network.negative_log_likelihood(batch)
            loss.backward()
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.gradient_clip_val)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        return total_loss / max(total_batches, 1)

    def _evaluate(self, loader: Any) -> float:
        assert self._network is not None and self._device is not None
        self._network.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                loss = self._network.negative_log_likelihood(batch)
                total_loss += float(loss.detach().cpu())
                total_batches += 1
        return total_loss / max(total_batches, 1)

    def fit(self, train_data: RollingWindowDataset, val_data: Optional[RollingWindowDataset] = None) -> None:
        self._validate_data(train_data)
        if val_data is not None:
            self._validate_data(val_data)
        if len(train_data) == 0:
            raise ValueError("train_data must contain at least one window.")

        require_torch()
        set_torch_seed(self.seed)
        self._asset_ids = list(train_data.asset_ids)
        self._prediction_length = train_data.prediction_length

        sample_loader = self._make_loader(train_data, shuffle=False)
        first_batch = next(iter_torch_batches(sample_loader, resolve_torch_device("cpu")))
        time_feature_dim = int(first_batch["past_time_feat"].shape[-1])
        self._build_network(train_data.num_assets, time_feature_dim)

        train_loader = self._make_loader(train_data, shuffle=True)
        val_loader = self._make_loader(val_data, shuffle=False) if val_data is not None and len(val_data) > 0 else None

        optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_score = float("inf")
        best_state = None
        bad_epochs = 0
        self.training_history = []
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._evaluate(val_loader) if val_loader is not None else train_loss
            self.training_history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            if self.verbose:
                print(f"DeepVAR epoch {epoch}/{self.max_epochs}: train={train_loss:.6f} val={val_loss:.6f}")

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

        loader = self._make_loader(test_data, shuffle=False)
        self._network.eval()
        sample_chunks = []
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self._device):
                sample_chunks.append(self._network.sample(batch, int(num_samples)).cpu().numpy())
        samples = np.concatenate(sample_chunks, axis=0)

        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "deepvar",
                "implementation": "finprobts_native_deepvar",
                "reference": "GluonTS DeepVAREstimator / VEC-LSTM",
                "seed": self.seed,
                "rank": self.rank,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "rnn_type": self.rnn_type,
                "lags_seq": list(self.lags_seq),
                "num_parallel_samples_default": self.num_parallel_samples,
                "cardinality": list(self.cardinality),
                "conditioning_length": self.conditioning_length,
                "use_marginal_transformation": self.use_marginal_transformation,
                "pick_incomplete": self.pick_incomplete,
                "use_time_features": self.use_time_features,
                "use_scale_feature": self.use_scale_feature,
                "model_internal_scaling": self.scaling,
                "training_prediction_length": self._prediction_length,
                "training_history": list(self.training_history),
            },
        )

    def save(self, path: str) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("Cannot save an unfitted DeepVARForecastModel.")
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "init_params": self._init_params(),
            "model_state": self._network.state_dict(),
            "num_assets": self._num_assets,
            "asset_ids": self._asset_ids,
            "prediction_length": self._prediction_length,
            "time_feature_dim": self._time_feature_dim,
            "is_fitted": self._is_fitted,
            "training_history": self.training_history,
        }
        torch.save(payload, output_dir / "model.pt")

    @classmethod
    def load(cls, path: str) -> "DeepVARForecastModel":
        require_torch()
        checkpoint_path = Path(path) / "model.pt"
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - old torch versions
            payload = torch.load(checkpoint_path, map_location="cpu")

        model = cls(**payload["init_params"])
        model._asset_ids = [str(item) for item in payload.get("asset_ids", [])]
        model._prediction_length = payload.get("prediction_length")
        time_feature_dim = int(payload.get("time_feature_dim", 4 if model.use_time_features else 0))
        model._build_network(int(payload["num_assets"]), time_feature_dim)
        model._network.load_state_dict(payload["model_state"])
        model._network.to(model._device)
        model._is_fitted = bool(payload.get("is_fitted", True))
        model.training_history = list(payload.get("training_history", []))
        return model
