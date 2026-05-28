"""
Case 4: Market-wide (shared) block-wise Markov regime switching panel AR(1).

Key features (as intended for the paper):
- One shared latent regime path s_t across all series (market-wide regimes).
- Regime is constant within blocks of length `block_size`.
- Block regimes follow a 3-state Markov chain with transition matrix Pi_block.
- Panel generation:
    y_{i,t} = a_i * mu_{s_t} + phi * y_{i,t-1} + b_i * sigma_{s_t} * eps_{i,t}
  where eps_{i,t} ~ N(0,1), and (a_i, b_i) provide mild cross-sectional heterogeneity.

Outputs:
- result["y"]: (T, N) panel
- result["states"]: (T,) shared regime labels in {0,1,2}
- result["params"]: full param dump + (a_i, b_i) vectors (JSON-serializable via lists)
- to_dataframe(): tidy long format: time, series_id, y, regime, regime_label
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .base import BaseSimulator


class MarketRegimePanelSimulator(BaseSimulator):
    """
    Market-wide regime switching panel AR(1), block-wise Markov switching.

    Regimes: 0=Up, 1=Stable, 2=Down
    Shared regime s_t across all series.

    y_{i,t} = a_i * mu_{s_t} + phi * y_{i,t-1} + b_i * sigma_{s_t} * eps_{i,t}
    """

    def __init__(
        self,
        T: int = 3000,
        n_firms: int = 50,
        block_size: int = 20,
        # Regime-specific means (market-level)
        mu_U: float = 0.001,
        mu_S: float = 0.0,
        mu_D: float = -0.001,
        # Regime-specific volatilities (market-level)
        sigma_U: float = 0.01,
        sigma_S: float = 0.008,
        sigma_D: float = 0.02,
        # AR(1) coefficient (shared across firms for isolation)
        phi: float = 0.1,
        # Block transition matrix (3x3). Rows sum to 1.
        Pi_block: Optional[np.ndarray] = None,
        # Cross-sectional heterogeneity controls (multiplicative scales around 1)
        # If set to 0, all a_i/b_i = 1.
        mu_scale_logsigma: float = 0.10,   # heterogeneity in mean loading a_i
        sig_scale_logsigma: float = 0.10,  # heterogeneity in volatility loading b_i
        # Burn-in
        burn_in: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)

        if T <= 0:
            raise ValueError("T must be positive.")
        if n_firms <= 0:
            raise ValueError("n_firms must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if not (-1.0 < phi < 1.0):
            raise ValueError("phi must be in (-1, 1) for AR(1) stability.")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0.")

        self.T = int(T)
        self.n_firms = int(n_firms)
        self.block_size = int(block_size)

        self.mu = np.array([mu_U, mu_S, mu_D], dtype=float)
        self.sigma = np.array([sigma_U, sigma_S, sigma_D], dtype=float)
        if np.any(self.sigma <= 0):
            raise ValueError("All regime sigmas must be > 0.")

        self.phi = float(phi)

        # Default transition if not provided
        if Pi_block is None:
            Pi_block = np.array(
                [
                    [0.92, 0.07, 0.01],  # from Up
                    [0.08, 0.84, 0.08],  # from Stable
                    [0.02, 0.08, 0.90],  # from Down
                ],
                dtype=float,
            )

        Pi_block = np.asarray(Pi_block, dtype=float)
        if Pi_block.shape != (3, 3):
            raise ValueError("Pi_block must be a 3x3 matrix.")
        if np.any(Pi_block < 0):
            raise ValueError("Pi_block must be nonnegative.")
        row_sums = Pi_block.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(f"Rows of Pi_block must sum to 1. Got {row_sums}.")
        self.Pi_block = Pi_block

        self.mu_scale_logsigma = float(mu_scale_logsigma)
        self.sig_scale_logsigma = float(sig_scale_logsigma)

        self.burn_in = int(burn_in)

    def _draw_lognormal_scale(self, n: int, logsigma: float) -> np.ndarray:
        """
        Draw multiplicative scales with E[scale]=1 using LogNormal(m, s^2) with m=-0.5*s^2.
        If logsigma==0 => all ones.
        """
        if logsigma <= 0.0:
            return np.ones(n, dtype=float)
        s = float(logsigma)
        m = -0.5 * s * s
        return self.rng.lognormal(mean=m, sigma=s, size=n).astype(float)

    def simulate(self) -> Dict[str, Any]:
        T_full = self.T + self.burn_in
        N = self.n_firms

        # ---- shared block regimes ----
        n_blocks = int(np.ceil(T_full / self.block_size))
        states_block = np.zeros(n_blocks, dtype=int)
        states_block[0] = 1  # start in Stable by default

        for b in range(1, n_blocks):
            prev = states_block[b - 1]
            states_block[b] = self.rng.choice(3, p=self.Pi_block[prev])

        states = np.repeat(states_block, self.block_size)[:T_full]  # (T_full,)

        # ---- cross-sectional scaling (mild heterogeneity) ----
        a_i = self._draw_lognormal_scale(N, self.mu_scale_logsigma)   # mean loading
        b_i = self._draw_lognormal_scale(N, self.sig_scale_logsigma)  # vol loading

        # ---- AR(1) panel generation ----
        eps = self.rng.normal(size=(T_full, N))  # iid across i,t
        y = np.zeros((T_full, N), dtype=float)

        for t in range(T_full):
            st = states[t]
            mu_t_i = a_i * self.mu[st]                # (N,)
            sig_t_i = b_i * self.sigma[st]            # (N,)

            if t == 0:
                y[t] = mu_t_i + sig_t_i * eps[t]
            else:
                y[t] = mu_t_i + self.phi * y[t - 1] + sig_t_i * eps[t]

        # drop burn-in
        y_keep = y[self.burn_in :]
        states_keep = states[self.burn_in :]

        self._simulation_result = {
            "y": y_keep,                 # (T, N)
            "states": states_keep,       # (T,)
            "a_i": a_i,                  # (N,)
            "b_i": b_i,                  # (N,)
            "params": {
                "case": "case4_market_regime_panel",
                "T": self.T,
                "n_firms": self.n_firms,
                "block_size": self.block_size,
                "mu": self.mu.tolist(),
                "sigma": self.sigma.tolist(),
                "phi": self.phi,
                "Pi_block": self.Pi_block.tolist(),
                "mu_scale_logsigma": self.mu_scale_logsigma,
                "sig_scale_logsigma": self.sig_scale_logsigma,
                "burn_in": self.burn_in,
                "seed": self.seed,
                # store scales for reproducibility / oracle diagnostics (JSON-friendly)
                "a_i": a_i.tolist(),
                "b_i": b_i.tolist(),
            },
        }
        return self._simulation_result

    def _convert_to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        regime_map = {0: "Up", 1: "Stable", 2: "Down"}
        y = result["y"]
        states = result["states"]
        T, N = y.shape

        # tidy long
        df = pd.DataFrame(
            {
                "time": np.repeat(np.arange(T), N),
                "series_id": np.tile(np.arange(N), T),
                "y": y.reshape(-1),
                "regime": np.repeat(states, N),
            }
        )
        df["regime_label"] = df["regime"].map(regime_map)
        return df
