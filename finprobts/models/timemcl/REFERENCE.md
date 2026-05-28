# TimeMCL Reference Notes

Reference:
- Native FinProbTS implementation of the TimeMCL multiple-choice forecasting design.

Paper:
- Adrien Cortes, Remi Rehm, Victor Letzelter. "Winner-Takes-All for Multivariate Probabilistic Time Series Forecasting." Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:11288-11312, 2025.
- arXiv: https://arxiv.org/abs/2506.05515
- PMLR: https://proceedings.mlr.press/v267/cortes25b.html

Upstream repo:
- https://github.com/Victorletzelter/timeMCL
- Local reference copy inspected at `C:\Users\Sun Jiaze\PycharmProjects\Prob_models\TimeMCL_new`.

License:
- Apache License 2.0 in the upstream repository.

Deviations from upstream:
- Keeps the FinProbTS benchmark contract: input is `RollingWindowDataset`; output is `ForecastResult` with samples shaped `[num_windows, num_samples, prediction_length, num_assets]`.
- Uses plain PyTorch loops instead of the upstream GluonTS/PyTorch Lightning estimator stack.
- Uses FinProbTS complete rolling windows instead of GluonTS `InstanceSplitter` sampling; `pick_incomplete=True` is not supported.
- The default FinProbTS task is one-step-ahead forecasting with a 96-step context window; model-level `context_length` and `prediction_length` are validated against the task when supplied.
- Implements the upstream default `backbone_deleted=True` TimeMCL head design with score heads, `min_ext_sum` / `min_in_sum`, and `wta` / `relaxed-wta` / `awta` losses.
- Uses FinProbTS calendar time features and optional target-dimension embeddings; custom upstream `time_features` wiring is deferred.
- Supports `mean` and `nops` scaling in this native version; upstream mean/std variants are noted but not wired into the FinProbTS path yet.
- Forecast samples are obtained by resampling learned hypotheses according to score heads so any requested `num_samples` can be returned from a finite hypothesis set.
