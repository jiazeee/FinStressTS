"""
Case 6: Zero-inflated jumps (ZIP counts) + signed jump magnitudes + AR(1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base import BaseSimulator


class ZeroInflatedJumpsSimulator(BaseSimulator):
    """
    AR(1) process with zero-inflated Poisson (ZIP) jump counts.

    DGP:
        y_t = mu + phi * y_{t-1} + eps_t + J_t
        eps_t ~ N(0, sigma_eps^2)

        N_t ~ ZIP(pi, lam)
          with prob pi: N_t = 0        (structural zeros)
          with prob 1-pi: N_t ~ Poisson(lam)

        J_t = sum_{i=1}^{N_t} s_{t,i} * kappa_{t,i}
          kappa_{t,i} ~ LogNormal(mean chosen to match E[kappa]=jump_mean_abs, sigma=jump_sigma_log)
          s_{t,i} in {+1,-1} with P(+1)=p_up
    """

    def __init__(
        self,
        T: int = 2000,
        mu: float = 0.0,
        phi: float = 0.1,
        sigma_eps: float = 0.01,
        pi: float = 0.9,
        lam: float = 0.2,
        jump_mean_abs: float = 0.03,
        jump_sigma_log: float = 0.6,
        p_up: float = 0.5,
        burn_in: int = 200,
        eps: float = 1e-12,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)

        if not (0.0 <= pi < 1.0):
            raise ValueError("pi must be in [0, 1).")
        if lam <= 0.0:
            raise ValueError("lam must be > 0.")
        if not (abs(phi) < 1.0):
            raise ValueError("Require |phi| < 1 for covariance-stationary AR(1).")
        if jump_mean_abs <= 0.0:
            raise ValueError("jump_mean_abs must be > 0.")
        if jump_sigma_log <= 0.0:
            raise ValueError("jump_sigma_log must be > 0.")
        if not (0.0 <= p_up <= 1.0):
            raise ValueError("p_up must be in [0, 1].")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0.")

        self.T = T
        self.mu = mu
        self.phi = phi
        self.sigma_eps = sigma_eps
        self.pi = pi
        self.lam = lam
        self.jump_mean_abs = jump_mean_abs
        self.jump_sigma_log = jump_sigma_log
        self.p_up = p_up
        self.burn_in = burn_in
        self.eps = eps

    # -------------------------
    # ZIP sampling
    # -------------------------
    def _sample_zip_counts(self, T: int) -> np.ndarray:
        """
        Sample ZIP counts:
          with prob pi -> 0
          else -> Poisson(lam)
        """
        structural_zero = self.rng.random(T) < self.pi
        counts = self.rng.poisson(lam=self.lam, size=T)
        counts[structural_zero] = 0
        return counts

    # -------------------------
    # Lognormal magnitude with specified mean
    # -------------------------
    def _draw_lognormal_with_mean(self, n: int) -> np.ndarray:
        """
        Draw LogNormal(m, s) such that E[exp(N(m,s^2))] = jump_mean_abs.
        For lognormal:
          mean = exp(m + 0.5 s^2)
        => m = log(mean) - 0.5 s^2
        """
        s = self.jump_sigma_log
        m = np.log(self.jump_mean_abs + self.eps) - 0.5 * s * s
        return self.rng.lognormal(mean=m, sigma=s, size=n)

    def simulate(self) -> Dict[str, Any]:
        T_full = self.T + self.burn_in

        # Jump counts (ZIP)
        N = self._sample_zip_counts(T_full)

        # Gaussian noise
        eps = self.rng.normal(0.0, self.sigma_eps, size=T_full)

        # Total jump per t
        J = np.zeros(T_full)

        # Realize jumps (avoid allocating T x maxN huge arrays)
        for t in range(T_full):
            k = int(N[t])
            if k > 0:
                mags = self._draw_lognormal_with_mean(k)
                signs = np.where(self.rng.random(k) < self.p_up, 1.0, -1.0)
                J[t] = float(np.sum(signs * mags))

        # AR(1) recursion
        y = np.zeros(T_full)
        for t in range(T_full):
            if t == 0:
                y[t] = self.mu + eps[t] + J[t]
            else:
                y[t] = self.mu + self.phi * y[t - 1] + eps[t] + J[t]

        # Drop burn-in
        y = y[self.burn_in :]
        N = N[self.burn_in :]
        J = J[self.burn_in :]

        # Useful closed-form moments for docs
        p0 = self.pi + (1.0 - self.pi) * np.exp(-self.lam)  # P(N=0)
        EN = (1.0 - self.pi) * self.lam

        self._simulation_result = {
            "y": y,
            "jump_count": N,
            "jump_size": J,
            "params": {
                "T": self.T,
                "mu": self.mu,
                "phi": self.phi,
                "sigma_eps": self.sigma_eps,
                "pi": self.pi,
                "lam": self.lam,
                "P_N_eq_0": float(p0),
                "E_N": float(EN),
                "jump_mean_abs": self.jump_mean_abs,
                "jump_sigma_log": self.jump_sigma_log,
                "p_up": self.p_up,
                "burn_in": self.burn_in,
            },
        }
        return self._simulation_result

    def _convert_to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        T = len(result["y"])
        return pd.DataFrame(
            {
                "time": np.arange(T),
                "series_id": 0,
                "y": result["y"],
                "jump_count": result["jump_count"],
                "jump_size": result["jump_size"],
            }
        )

"""
Case 6 Option A (market-wide ZIP jumps) — PANEL VERSION

Key idea (Option A):
  - One common ZIP jump-count process N_t for the whole market.
  - One common market jump aggregate J_t (compound signed lognormal).
  - Panel observations: each firm i has exposure gamma_i to the common jump:
        y_{i,t} = alpha_i + beta_i' f_t + phi y_{i,t-1} + eps_{i,t} + gamma_i * J_t

This yields panel data with cross-sectional dependence via:
  - common factors f_t (optional; set n_factors=0 to disable)
  - common jump component gamma_i * J_t
"""



class MarketZIPPanelSimulator(BaseSimulator):
    """
    Market-wide Zero-Inflated Poisson (ZIP) jumps + panel observation (Option A).

    Common market jump process:
      N_t ~ ZIP(pi, lam)
      J_t = sum_{k=1..N_t} s_{t,k} * a_{t,k}
        s_{t,k} in {+1,-1}, P(+1)=p_up
        a_{t,k} ~ LogNormal(m_log, s_log^2), with E[a]=jump_mean_abs

    Panel observation:
      y_{i,t} = alpha_i + beta_i' f_t + phi * y_{i,t-1} + eps_{i,t} + gamma_i * J_t
      eps_{i,t} ~ N(0, sigma_eps^2)

    Factors (optional):
      f_t ~ N(mu_f, sigma_f^2 I_K)
      Set n_factors=0 to disable the factor term.
    """

    def __init__(
        self,
        T: int = 2000,
        n_firms: int = 100,
        n_factors: int = 3,          # set to 0 to disable factors
        # AR(1) + diffusive noise
        phi: float = 0.1,
        sigma_eps: float = 0.01,
        # ZIP jump counts (market-wide)
        pi: float = 0.90,            # structural zero prob
        lam: float = 0.20,           # Poisson rate when not a structural zero
        # Jump magnitudes
        jump_mean_abs: float = 0.03, # E[a] for lognormal magnitudes
        jump_sigma_log: float = 0.60,
        p_up: float = 0.50,          # sign probability
        # Exposure and cross-sectional params
        gamma_mean: float = 1.0,
        gamma_std: float = 0.2,
        alpha_i_std: float = 0.001,
        beta_mean: float = 0.5,
        beta_std: float = 0.2,
        # Factor params (only used if n_factors>0)
        mu_f: float = 0.0,
        sigma_f: float = 0.01,
        # Burn-in
        burn_in: int = 200,
        eps: float = 1e-12,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)

        if T <= 0:
            raise ValueError("T must be positive.")
        if n_firms <= 0:
            raise ValueError("n_firms must be positive.")
        if n_factors < 0:
            raise ValueError("n_factors must be >= 0.")
        if not (abs(phi) < 1.0):
            raise ValueError("Require |phi| < 1 for covariance-stationary AR(1).")
        if sigma_eps <= 0:
            raise ValueError("sigma_eps must be > 0.")
        if not (0.0 <= pi < 1.0):
            raise ValueError("pi must be in [0, 1).")
        if lam <= 0.0:
            raise ValueError("lam must be > 0.")
        if jump_mean_abs <= 0.0:
            raise ValueError("jump_mean_abs must be > 0.")
        if jump_sigma_log <= 0.0:
            raise ValueError("jump_sigma_log must be > 0.")
        if not (0.0 <= p_up <= 1.0):
            raise ValueError("p_up must be in [0, 1].")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0.")
        if gamma_std < 0:
            raise ValueError("gamma_std must be >= 0.")
        if sigma_f <= 0:
            raise ValueError("sigma_f must be > 0 (if using factors).")

        self.T = int(T)
        self.n_firms = int(n_firms)
        self.n_factors = int(n_factors)

        self.phi = float(phi)
        self.sigma_eps = float(sigma_eps)

        self.pi = float(pi)
        self.lam = float(lam)

        self.jump_mean_abs = float(jump_mean_abs)
        self.jump_sigma_log = float(jump_sigma_log)
        self.p_up = float(p_up)

        self.gamma_mean = float(gamma_mean)
        self.gamma_std = float(gamma_std)

        self.alpha_i_std = float(alpha_i_std)
        self.beta_mean = float(beta_mean)
        self.beta_std = float(beta_std)

        self.mu_f = float(mu_f)
        self.sigma_f = float(sigma_f)

        self.burn_in = int(burn_in)
        self.eps = float(eps)

    # -------------------------
    # ZIP sampling (market-wide)
    # -------------------------
    def _sample_zip_counts(self, T: int) -> np.ndarray:
        """
        Sample ZIP counts N_t:
          with prob pi -> 0
          else -> Poisson(lam)
        """
        structural_zero = self.rng.random(T) < self.pi
        counts = self.rng.poisson(lam=self.lam, size=T)
        counts[structural_zero] = 0
        return counts

    # -------------------------
    # Lognormal magnitude with specified mean
    # -------------------------
    def _draw_lognormal_with_mean(self, n: int) -> np.ndarray:
        """
        Draw LogNormal(m, s) such that E[exp(N(m,s^2))] = jump_mean_abs.
        For lognormal mean: exp(m + 0.5 s^2) => m = log(mean) - 0.5 s^2
        """
        s = self.jump_sigma_log
        m = np.log(self.jump_mean_abs + self.eps) - 0.5 * s * s
        return self.rng.lognormal(mean=m, sigma=s, size=n)

    def simulate(self) -> Dict[str, Any]:
        T_full = self.T + self.burn_in

        # -------------------------
        # Common market jump process
        # -------------------------
        N = self._sample_zip_counts(T_full)          # (T_full,)
        J = np.zeros(T_full, dtype=float)            # (T_full,)

        for t in range(T_full):
            k = int(N[t])
            if k > 0:
                mags = self._draw_lognormal_with_mean(k)
                signs = np.where(self.rng.random(k) < self.p_up, 1.0, -1.0)
                J[t] = float(np.sum(signs * mags))

        # -------------------------
        # Optional factors
        # -------------------------
        if self.n_factors > 0:
            f = self.rng.normal(
                loc=self.mu_f,
                scale=self.sigma_f,
                size=(T_full, self.n_factors),
            )
        else:
            f = np.zeros((T_full, 0), dtype=float)

        # -------------------------
        # Cross-sectional parameters
        # -------------------------
        alpha_i = self.rng.normal(0.0, self.alpha_i_std, size=self.n_firms)
        gamma_i = self.rng.normal(self.gamma_mean, self.gamma_std, size=self.n_firms)

        if self.n_factors > 0:
            beta_i = self.rng.normal(
                loc=self.beta_mean,
                scale=self.beta_std,
                size=(self.n_firms, self.n_factors),
            )  # shape (N, K)
        else:
            beta_i = np.zeros((self.n_firms, 0), dtype=float)

        # -------------------------
        # Panel AR(1) observation
        # -------------------------
        eps = self.rng.normal(0.0, self.sigma_eps, size=(T_full, self.n_firms))
        y = np.zeros((T_full, self.n_firms), dtype=float)

        for t in range(T_full):
            factor_term = (f[t] @ beta_i.T) if self.n_factors > 0 else 0.0  # shape (N,)
            jump_term = gamma_i * J[t]                                      # shape (N,)

            if t == 0:
                y[t] = alpha_i + factor_term + eps[t] + jump_term
            else:
                y[t] = alpha_i + factor_term + self.phi * y[t - 1] + eps[t] + jump_term

        # -------------------------
        # Drop burn-in
        # -------------------------
        y = y[self.burn_in :]
        N = N[self.burn_in :]
        J = J[self.burn_in :]
        f = f[self.burn_in :]

        # Useful closed-form moments for docs
        p0 = self.pi + (1.0 - self.pi) * np.exp(-self.lam)     # P(N=0)
        EN = (1.0 - self.pi) * self.lam                         # E[N]
        EJ_abs_approx = EN * self.jump_mean_abs                 # approx E[|J|] if signs ~ symmetric

        self._simulation_result = {
            "y": y,                           # (T, N)
            "factors": f,                     # (T, K) or (T,0)
            "jump_count": N,                  # (T,)
            "jump_size": J,                   # (T,)
            "gamma": gamma_i,                 # (N,)
            "alpha_i": alpha_i,               # (N,)
            "beta_i": beta_i,                 # (N,K) or (N,0)
            "params": {
                "T": self.T,
                "burn_in": self.burn_in,
                "n_firms": self.n_firms,
                "n_factors": self.n_factors,
                "phi": self.phi,
                "sigma_eps": self.sigma_eps,
                "pi": self.pi,
                "lam": self.lam,
                "P_N_eq_0": float(p0),
                "E_N": float(EN),
                "E_abs_J_approx": float(EJ_abs_approx),
                "jump_mean_abs": self.jump_mean_abs,
                "jump_sigma_log": self.jump_sigma_log,
                "p_up": self.p_up,
                "gamma_mean": self.gamma_mean,
                "gamma_std": self.gamma_std,
                "alpha_i_std": self.alpha_i_std,
                "beta_mean": self.beta_mean,
                "beta_std": self.beta_std,
                "mu_f": self.mu_f,
                "sigma_f": self.sigma_f,
                "eps": self.eps,
            },
        }
        return self._simulation_result

    def _convert_to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """
        Tidy long panel:
          time, series_id, y,
          jump_count (common), jump_size (common), is_jump (common),
          gamma_i (firm exposure)

        (Optionally) factor columns f1..fK if n_factors>0.
        """
        y = result["y"]
        N = result["jump_count"]
        J = result["jump_size"]
        gamma = result["gamma"]
        f = result.get("factors", None)

        T, n_firms = y.shape
        is_jump = (N > 0)

        rows = []
        for i in range(n_firms):
            d = {
                "time": np.arange(T),
                "series_id": i,
                "y": y[:, i],
                "jump_count": N,
                "jump_size": J,
                "is_jump": is_jump,
                "gamma": np.full(T, gamma[i], dtype=float),
            }
            if f is not None and f.shape[1] > 0:
                for k in range(f.shape[1]):
                    d[f"f{k+1}"] = f[:, k]
            rows.append(pd.DataFrame(d))

        return pd.concat(rows, ignore_index=True)
