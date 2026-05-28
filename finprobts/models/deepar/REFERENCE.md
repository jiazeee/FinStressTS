# DeepAR Reference Note

Reference:
- Paper: Salinas, Flunkert, and Gasthaus, "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", arXiv:1704.04110 / International Journal of Forecasting.
- Upstream repo: GluonTS `DeepAREstimator`, https://github.com/awslabs/gluonts.
- License: GluonTS is Apache-2.0. This FinProbTS implementation is a native PyTorch implementation and does not copy GluonTS source code.
- Deviations from upstream: Uses the FinProbTS `RollingWindowDataset` as the public input contract; treats each asset as one related univariate series and reassembles marginal samples into FinProbTS `[windows, samples, horizon, assets]` output; uses FinProbTS calendar features plus a simple log-age feature rather than the full GluonTS transformation stack; uses fixed complete rolling windows instead of GluonTS instance samplers; uses a plain PyTorch training loop instead of PyTorch Lightning; supports Student-t and Normal output names but not arbitrary GluonTS `DistributionOutput` objects yet; external dynamic/static feature fields, custom imputation, custom time features, and custom samplers are not wired in yet.
