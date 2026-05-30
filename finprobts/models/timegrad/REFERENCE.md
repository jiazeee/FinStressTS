# TimeGrad Reference Note

Reference:
- Native FinProbTS implementation of the PyTorchTS TimeGrad estimator design.

Paper:
- Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf, "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", Proceedings of the 38th International Conference on Machine Learning, PMLR 139:8857-8868, 2021 / arXiv:2101.12072.
- PMLR: https://proceedings.mlr.press/v139/rasul21a.html

Upstream repo:
- https://github.com/zalandoresearch/pytorch-ts
- Upstream files used as architectural references:
  - `pts/model/time_grad/time_grad_estimator.py`
  - `pts/model/time_grad/time_grad_network.py`
  - `pts/model/time_grad/epsilon_theta.py`
  - `pts/modules/gaussian_diffusion.py`

License:
- PyTorchTS is published with MIT and Apache-2.0 license files in the upstream repository. This FinProbTS implementation is a clean native PyTorch implementation rather than vendored upstream source.

Deviations from upstream:
- FinProbTS consumes the canonical `RollingWindowDataset` directly instead of GluonTS `ListDataset`, transformations, and `InstanceSplitter`.
- Forecast output is standardized as `ForecastResult.samples` with shape `[num_windows, num_samples, prediction_length, num_assets]`.
- The rolling context window is treated as the available history window. Upstream constructs `history_length = context_length + max(lags_seq)` internally through its splitter.
- Time features use FinProbTS calendar features, not GluonTS Fourier time features. Custom `time_features` are rejected for now.
- `pick_incomplete=True` is not implemented because FinProbTS currently uses complete rolling windows.
- The Gaussian diffusion and epsilon network follow the PyTorchTS architecture and DDPM equations, but the code is written natively for the FinProbTS batch format and exposes per-window weighted losses.
- The default benchmark task is still 1-step-ahead with context length 96; the native decoder supports recursive multi-step sampling when the task horizon is larger.
