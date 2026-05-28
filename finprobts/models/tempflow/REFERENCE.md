# TempFlow Reference Note

Reference:
- Native FinProbTS implementation of the PyTorchTS TempFlow estimator design.

Paper:
- Kashif Rasul, Abdul-Saboor Sheikh, Ingmar Schuster, Urs Bergmann, Roland Vollgraf, "Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows", ICLR 2021.

Upstream repo:
- https://github.com/zalandoresearch/pytorch-ts
- Upstream files used as architectural references:
  - `pts/model/tempflow/tempflow_estimator.py`
  - `pts/model/tempflow/tempflow_network.py`
  - `pts/modules/flows.py`

License:
- PyTorchTS is published with MIT and Apache-2.0 license files in the upstream repository. This FinProbTS implementation is a clean native PyTorch implementation rather than vendored upstream source.

Deviations from upstream:
- FinProbTS consumes the canonical `RollingWindowDataset` directly instead of GluonTS `ListDataset`, transformations, and `InstanceSplitter`.
- Forecast output is standardized as `ForecastResult.samples` with shape `[num_windows, num_samples, prediction_length, num_assets]`.
- The active implementation supports the upstream default `flow_type="RealNVP"`; upstream also exposes `MAF`.
- The rolling context window is treated as the available history window. Upstream constructs `history_length = context_length + max(lags_seq)` internally through its splitter.
- Time features use FinProbTS calendar features, not GluonTS Fourier time features. Custom `time_features` are rejected for now.
- `dequantize=True` and `pick_incomplete=True` are not implemented because FinProbTS currently uses continuous, complete rolling windows.
- The normalizing flow is reimplemented with conditional affine coupling layers and an isotropic normal base distribution; it follows the same modeling contract but is not a byte-for-byte copy of PyTorchTS flow modules.
