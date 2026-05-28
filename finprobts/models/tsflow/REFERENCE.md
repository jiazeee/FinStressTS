# TSFlow Reference Notes

Reference:
- Native FinProbTS implementation of the conditional TSFlow forecasting design.

Paper:
- Marcel Kollovieh, Marten Lienen, David Luedke, Leo Schwinn, Stephan Guennemann. "Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting." ICLR 2025.
- arXiv: https://arxiv.org/abs/2410.03024
- ICLR: https://proceedings.iclr.cc/paper_files/paper/2025/hash/ee1a1ecc92f35702b5c29dad3dc909ea-Abstract-Conference.html

Upstream repo:
- https://github.com/marcelkollovieh/TSFlow
- Local reference copy inspected at `C:\Users\Sun Jiaze\PycharmProjects\Prob_models\TSFlow`.

License:
- No license file was found in the inspected local/public TSFlow repository. For that reason this FinProbTS version does not vendor TSFlow source files; it implements the paper/repo architecture natively with PyTorch primitives.

Deviations from upstream:
- Keeps the FinProbTS benchmark contract: input is `RollingWindowDataset`; output is `ForecastResult` with samples shaped `[num_windows, num_samples, prediction_length, num_assets]`.
- Uses plain PyTorch loops instead of the upstream Lightning/GluonTS/PyTorchPredictor training and inference stack.
- Implements the conditional TSFlow path with Gaussian-process regression priors, random conditional flow matching, lag features, observation-mask features, sinusoidal flow-time embeddings, and Euler sampling.
- Replaces the upstream S4 and `linear_attention_transformer` blocks with native PyTorch temporal and cross-asset transformer residual blocks to keep FinProbTS dependencies minimal.
- Does not implement optimal-transport pairing, NeuralODE solvers beyond Euler, unconditional/PS variants, or guided quantile sampling yet.
- Filters upstream calendar lags to those available inside the FinProbTS context window; the default benchmark task remains one-step-ahead with a 96-step context.
