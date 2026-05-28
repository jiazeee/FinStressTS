"""
Case 5 (Option A): Market-wide self-exciting jumps + panel observations.

What this simulator generates
-----------------------------
A panel {y_{i,t}} with cross-sectional dependence driven by:
  (i) common Hawkes jump process J_t (market-wide clustered jumps),
 (ii) optional common Gaussian factors f_t with firm loadings beta_i,
(iii) firm-specific AR(1) dynamics + idiosyncratic Gaussian noise.

DGP (kept series, after burn-in):
  y_{i,t} = alpha_i + phi * y_{i,t-1} + beta_i' f_t + eps_{i,t} + gamma_i * J_t

Common jump process:
  λ_t = μ + exp(-β) * (λ_{t-1} - μ) + α * N_{t-1}
  N_t ~ Poisson(λ_t)
  J_t = sum_{k=1..N_t} s_{t,k} * A_{t,k},  s in {+1,-1},  A lognormal with mean jump_mean_abs

Notes
-----
- Stability condition (discrete-time Hawkes recursion):
    alpha < 1 - exp(-beta)
- Jump magnitudes are lognormal with controlled mean jump_mean_abs.
- gamma_i (firm exposure to market jumps) is positive by default.
- Output "returns" is shape (T, n_firms). DataFrame is tidy long:
    time, series_id, y, event_count, intensity, jump_size
  with event_count/intensity/jump_size repeated across series_id at the same time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base import BaseSimulator


class MarketHawkesPanelSimulator(BaseSimulator):
    def __init__(
        self,
        # Panel size
        T: int = 2000,
        n_firms: int = 50,

        # Common Gaussian factor block (optional)
        n_factors: int = 3,
        factor_sigma: float = 0.01,   # std of each factor (iid Gaussian)
        beta_mean: float = 0.5,
        beta_std: float = 0.2,

        # Firm-level AR(1) + idio noise
        phi: float = 0.1,
        sigma_eps: float = 0.01,      # idiosyncratic innovation std
        alpha_i_std: float = 0.001,   # firm intercept dispersion

        # Market-wide Hawkes jump process
        mu: float = 0.05,             # baseline intensity (events per time step)
        alpha: float = 0.1,           # self-excitation from last count
        beta: float = 1.0,            # decay speed (through delta=exp(-beta))

        # Jump size distribution (per event)
        jump_mean_abs: float = 0.03,  # target mean absolute magnitude per event (before sign)
        jump_sigma_log: float = 0.6,  # lognormal sigma
        p_up: float = 0.5,            # probability of positive event sign

        # Firm exposure to market jump
        gamma_mean: float = 1.0,
        gamma_logsigma: float = 0.0,  # 0 => all firms same exposure (gamma_i = gamma_mean)

        # Simulation controls
        burn_in: int = 200,
        eps: float = 1e-8,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)

        # --- basic checks ---
        if T <= 0:
            raise ValueError("T must be positive.")
        if n_firms <= 0:
            raise ValueError("n_firms must be positive.")
        if n_factors < 0:
            raise ValueError("n_factors must be >= 0.")
        if not (-1.0 < phi < 1.0):
            raise ValueError("phi must be in (-1, 1) for AR(1) stability.")
        if sigma_eps <= 0:
            raise ValueError("sigma_eps must be > 0.")
        if factor_sigma < 0:
            raise ValueError("factor_sigma must be >= 0.")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0.")
        if mu < 0:
            raise ValueError("mu must be >= 0.")
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if beta <= 0:
            raise ValueError("beta must be > 0.")
        if jump_mean_abs <= 0:
            raise ValueError("jump_mean_abs must be > 0.")
        if jump_sigma_log < 0:
            raise ValueError("jump_sigma_log must be >= 0.")
        if not (0.0 <= p_up <= 1.0):
            raise ValueError("p_up must be in [0, 1].")
        if gamma_mean <= 0:
            raise ValueError("gamma_mean must be > 0.")
        if gamma_logsigma < 0:
            raise ValueError("gamma_logsigma must be >= 0.")

        self.T = int(T)
        self.n_firms = int(n_firms)

        self.n_factors = int(n_factors)
        self.factor_sigma = float(factor_sigma)
        self.beta_mean = float(beta_mean)
        self.beta_std = float(beta_std)

        self.phi = float(phi)
        self.sigma_eps = float(sigma_eps)
        self.alpha_i_std = float(alpha_i_std)

        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.jump_mean_abs = float(jump_mean_abs)
        self.jump_sigma_log = float(jump_sigma_log)
        self.p_up = float(p_up)

        self.gamma_mean = float(gamma_mean)
        self.gamma_logsigma = float(gamma_logsigma)

        self.burn_in = int(burn_in)
        self.eps = float(eps)

        # --- Hawkes stability check ---
        delta = float(np.exp(-self.beta))
        if self.alpha >= (1.0 - delta):
            raise ValueError(
                "Unstable Hawkes recursion: require alpha < 1 - exp(-beta). "
                f"Got alpha={self.alpha:.4f}, 1-exp(-beta)={(1.0-delta):.4f}."
            )

    def _draw_lognormal_with_mean(self, n: int) -> np.ndarray:
        """
        Draw lognormal magnitudes A with E[A] = jump_mean_abs.
        If ln(A) ~ N(m, s^2), then E[A] = exp(m + 0.5 s^2).
        So m = log(mean) - 0.5 s^2.
        """
        s = self.jump_sigma_log
        m = np.log(self.jump_mean_abs + 1e-12) - 0.5 * s * s
        return self.rng.lognormal(mean=m, sigma=s, size=n)

    def _draw_gamma_exposure(self) -> np.ndarray:
        """
        Positive firm exposures gamma_i.
        If gamma_logsigma=0 => constant gamma_mean.
        Else gamma_i = gamma_mean * exp( sigma * z - 0.5*sigma^2 ) so E[gamma_i]=gamma_mean.
        """
        if self.gamma_logsigma <= 0.0:
            return np.full(self.n_firms, self.gamma_mean, dtype=float)
        s = self.gamma_logsigma
        z = self.rng.normal(size=self.n_firms)
        return self.gamma_mean * np.exp(s * z - 0.5 * s * s)

    def simulate(self) -> Dict[str, Any]:
        T_full = self.T + self.burn_in
        delta = float(np.exp(-self.beta))

        # --- common Hawkes state ---
        lam = np.zeros(T_full, dtype=float)
        N = np.zeros(T_full, dtype=int)
        J = np.zeros(T_full, dtype=float)

        lam[0] = max(self.mu, self.eps)

        # --- common factors (optional) ---
        if self.n_factors > 0 and self.factor_sigma > 0:
            f = self.rng.normal(loc=0.0, scale=self.factor_sigma, size=(T_full, self.n_factors))
        else:
            f = np.zeros((T_full, self.n_factors), dtype=float)

        # --- cross-sectional params ---
        alpha_i = self.rng.normal(loc=0.0, scale=self.alpha_i_std, size=self.n_firms)
        if self.n_factors > 0:
            beta_i = self.rng.normal(loc=self.beta_mean, scale=self.beta_std, size=(self.n_firms, self.n_factors))
        else:
            beta_i = np.zeros((self.n_firms, 0), dtype=float)

        gamma_i = self._draw_gamma_exposure()

        # --- idiosyncratic eps and panel output ---
        eps_i = self.rng.normal(loc=0.0, scale=self.sigma_eps, size=(T_full, self.n_firms))
        y = np.zeros((T_full, self.n_firms), dtype=float)

        for t in range(T_full):
            if t > 0:
                lam[t] = self.mu + delta * (lam[t - 1] - self.mu) + self.alpha * N[t - 1]
                lam[t] = max(lam[t], self.eps)

            # event count and jump aggregation
            N[t] = self.rng.poisson(lam[t])

            if N[t] > 0:
                mags = self._draw_lognormal_with_mean(N[t])
                signs = np.where(self.rng.random(N[t]) < self.p_up, 1.0, -1.0)
                J[t] = float(np.sum(signs * mags))
            else:
                J[t] = 0.0

            # panel observation
            common_jump_term = gamma_i * J[t]  # shape (n_firms,)
            factor_term = (f[t] @ beta_i.T) if self.n_factors > 0 else 0.0  # shape (n_firms,) or scalar

            if t == 0:
                y[t] = alpha_i + factor_term + eps_i[t] + common_jump_term
            else:
                y[t] = alpha_i + self.phi * y[t - 1] + factor_term + eps_i[t] + common_jump_term

        # drop burn-in
        sl = slice(self.burn_in, T_full)
        y = y[sl]
        f = f[sl]
        lam = lam[sl]
        N = N[sl]
        J = J[sl]

        self._simulation_result = {
            "returns": y,               # (T, n_firms)
            "factors": f,               # (T, K)
            "event_count": N,           # (T,)
            "intensity": lam,           # (T,)
            "jump_size": J,             # (T,)
            "gamma": gamma_i,           # (n_firms,)
            "params": {
                "T": self.T,
                "burn_in": self.burn_in,
                "n_firms": self.n_firms,
                "n_factors": self.n_factors,
                "factor_sigma": self.factor_sigma,
                "beta_mean": self.beta_mean,
                "beta_std": self.beta_std,
                "phi": self.phi,
                "sigma_eps": self.sigma_eps,
                "alpha_i_std": self.alpha_i_std,
                "mu": self.mu,
                "alpha": self.alpha,
                "beta": self.beta,
                "delta": float(np.exp(-self.beta)),
                "branching_ratio_discrete": self.alpha / (1.0 - np.exp(-self.beta)),
                "jump_mean_abs": self.jump_mean_abs,
                "jump_sigma_log": self.jump_sigma_log,
                "p_up": self.p_up,
                "gamma_mean": self.gamma_mean,
                "gamma_logsigma": self.gamma_logsigma,
                "eps_floor": self.eps,
                "seed": self.seed,
            },
        }
        return self._simulation_result

    def _convert_to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        y = result["returns"]
        T, n_firms = y.shape

        lam = result["intensity"]
        N = result["event_count"]
        J = result["jump_size"]

        dfs = []
        for i in range(n_firms):
            dfs.append(
                pd.DataFrame(
                    {
                        "time": np.arange(T, dtype=int),
                        "series_id": i,
                        "y": y[:, i],
                        # common jump process diagnostics (repeated across series)
                        "event_count": N,
                        "intensity": lam,
                        "jump_size": J,
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)
