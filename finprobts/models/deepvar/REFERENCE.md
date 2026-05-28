# DeepVAR Reference Note

Reference:
- Paper: Salinas et al., "High-dimensional multivariate forecasting with low-rank Gaussian Copula Processes" / VEC-LSTM family, arXiv:1910.03002.
- Upstream repo: GluonTS `DeepVAREstimator`, https://github.com/awslabs/gluonts; frozen multivariate release referenced by GluonTS, https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release.
- License: GluonTS is Apache-2.0. This FinProbTS implementation is a native PyTorch implementation and does not copy GluonTS source code.
- Deviations from upstream: Uses the FinProbTS `RollingWindowDataset` as the public input contract; treats the provided context window as the available lag history instead of using GluonTS `history_length = context_length + max(lags_seq)` instance splitting; uses FinProbTS calendar features rather than GluonTS Fourier frequency features; uses a plain PyTorch training loop instead of MXNet/GluonTS trainers; `predict(..., num_samples=...)` controls the number of returned samples instead of the upstream `num_parallel_samples` predictor setting; omits GluonTS marginal CDF-to-Gaussian transformation, custom distribution outputs, custom time features, custom samplers, incomplete-window sampling, and non-default static categorical cardinalities for now.
