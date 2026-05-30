# RATD Reference Notes

Reference:
- Native FinProbTS implementation of Retrieval-Augmented Time series Diffusion.

Paper:
- Jingwei Liu, Ling Yang, Hongyan Li, Shenda Hong. "Retrieval-Augmented Diffusion Models for Time Series Forecasting." NeurIPS 2024.
- arXiv: https://arxiv.org/abs/2410.18712
- NeurIPS: https://papers.nips.cc/paper_files/paper/2024/hash/053ee34c0971568bfa5c773015c10502-Abstract-Conference.html

Upstream repo:
- https://github.com/stanliu96/RATD

License:
- MIT License in the upstream repository. The public RATD code states it is based on CSDI.

Deviations from upstream:
- Keeps the FinProbTS benchmark contract: input is `RollingWindowDataset`; output is `ForecastResult` with samples shaped `[num_windows, num_samples, prediction_length, num_assets]`.
- Uses plain PyTorch loops instead of the upstream experiment scripts and dataset-specific loaders.
- Uses FinProbTS rolling windows and in-memory retrieval over the training split instead of the paper's dataset-specific TCN checkpoint and saved retrieval index files.
- The retrieval encoder is currently normalized flattened context windows, matching the local `precompute_retrieval_idx.py` path. During training, exact same-window retrieval is excluded by window index, matching the official precomputed-index behavior more closely than score-threshold exclusion.
- Implements the RATD forecasting-as-imputation path with conditional masks over `[context + prediction_length]`, sinusoidal time embeddings, asset embeddings, diffusion-step embeddings, residual time/feature transformer blocks, and reference-modulated cross-asset attention.
- The reference-modulated attention block now follows the upstream width (`heads=8`, `dim_head=64`) and fixed transformer feed-forward width (`dim_feedforward=64`) rather than a reduced FinProbTS-specific attention width.
- Supports arbitrary FinProbTS prediction lengths, though the default benchmark task remains one-step-ahead with context length 96.
- Passes retrieved references during both training and sampling; the local upstream copy only passes references clearly through the training call path.
