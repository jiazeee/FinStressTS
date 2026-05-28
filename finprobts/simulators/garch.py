"""
GARCH volatility clustering simulator (Case 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base import BaseSimulator


class GARCHSimulator(BaseSimulator):
    """
    Simulate panel returns with a factor structure and GARCH(1,1) volatility
    dynamics in both factors and idiosyncratic residuals.

    DGP:
      Factors (j=1..K):
        f_{j,t} = mu_{f,j} + sqrt(sigma^2_{f,j,t}) * eps_{j,t}
        sigma^2_{f,j,t} = omega_{f,j} + a_f * (f_{j,t-1}-mu_{f,j})^2 + b_f * sigma^2_{f,j,t-1}

      Idiosyncratic residuals (i=1..N):
        u_{i,t} = sqrt(sigma^2_{u,i,t}) * eta_{i,t}
        sigma^2_{u,i,t} = omega_{u,i} + a_u * u_{i,t-1}^2 + b_u * sigma^2_{u,i,t-1}

      Returns:
        r_{i,t} = alpha_i + beta_i^T f_t + u_{i,t}

    Notes:
      - Conditional idiosyncratic shocks are independent across firms (diagonal idio cov),
        but returns are cross-correlated through the common factors.
      - omega terms are set so unconditional variances match target baselines when stationary.
      - Uses a local RNG from BaseSimulator for reproducibility.

    Difficulty knobs (recommended):
      - persistence_f = a_f + b_f, persistence_u = a_u + b_u
      - idio variance heterogeneity (lognormal sigma)
      - factor strength via sigma2_bar_factor and beta_i scale
    """

    def __init__(
        self,
        T: int = 2000,
        n_firms: int = 50,
        n_factors: int = 3,
        # Separate persistence controls (alpha+beta) for factors and idiosyncratic
        rho_f: float = 0.90,
        rho_u: float = 0.90,
        # Split rule for alpha/beta; alpha_share * rho => alpha, remainder => beta
        alpha_share_f: float = 0.10,
        alpha_share_u: float = 0.10,
        # Unconditional variance baselines
        sigma2_bar_factor: float = 0.02,
        sigma2_bar_idio: float = 0.04,
        # Cross-sectional heterogeneity for idio unconditional variances:
        # sigma_log = 0 => all firms have same sigma2_bar_idio
        idio_sigma_log: float = 0.30,
        # Factor mean (often 0 for returns-like factors)
        mu_f: float = 0.0,
        # Cross-sectional parameters for alpha_i, beta_i
        alpha_i_std: float = 0.001,
        beta_mean: float = 0.5,
        beta_std: float = 0.2,
        # Burn-in & numeric stability
        burn_in: int = 200,
        eps: float = 1e-10,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        if T <= 0:
            raise ValueError("T must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if n_firms <= 0 or n_factors <= 0:
            raise ValueError("n_firms and n_factors must be positive.")
        if not (0.0 <= alpha_share_f <= 1.0 and 0.0 <= alpha_share_u <= 1.0):
            raise ValueError("alpha_share_f/u must be in [0,1].")
        if rho_f >= 1.0 or rho_u >= 1.0:
            raise ValueError("rho_f and rho_u must be < 1 for covariance stationarity.")
        if rho_f <= 0.0 or rho_u <= 0.0:
            raise ValueError("rho_f and rho_u must be > 0.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.T = T
        self.n_firms = n_firms
        self.n_factors = n_factors

        self.rho_f = rho_f
        self.rho_u = rho_u
        self.alpha_share_f = alpha_share_f
        self.alpha_share_u = alpha_share_u

        self.sigma2_bar_factor = sigma2_bar_factor
        self.sigma2_bar_idio = sigma2_bar_idio
        self.idio_sigma_log = idio_sigma_log

        self.mu_f = mu_f
        self.alpha_i_std = alpha_i_std
        self.beta_mean = beta_mean
        self.beta_std = beta_std

        self.burn_in = burn_in
        self.eps = eps

    def simulate(self) -> Dict[str, Any]:
        # ---- GARCH params (split alpha/beta) ----
        a_f = self.alpha_share_f * self.rho_f
        b_f = self.rho_f - a_f
        a_u = self.alpha_share_u * self.rho_u
        b_u = self.rho_u - a_u

        # ---- Unconditional variances ----
        # Factors: same unconditional variance per factor (can be extended to per-factor heterogeneity)
        sigma2_bar_f = np.full(self.n_factors, float(self.sigma2_bar_factor))

        # Idiosyncratic: heterogeneous unconditional variances across firms (lognormal)
        if self.idio_sigma_log <= 0.0:
            sigma2_bar_u = np.full(self.n_firms, float(self.sigma2_bar_idio))
        else:
            # Lognormal with mean approximately sigma2_bar_idio
            # Let X ~ LogNormal(m, s^2). E[X] = exp(m + 0.5 s^2). Choose m accordingly.
            s = float(self.idio_sigma_log)
            m = np.log(float(self.sigma2_bar_idio)) - 0.5 * s * s
            sigma2_bar_u = self.rng.lognormal(mean=m, sigma=s, size=self.n_firms)

        # ---- omega to match unconditional variance under stationarity ----
        omega_f = (1.0 - (a_f + b_f)) * sigma2_bar_f  # vector length K
        omega_u = (1.0 - (a_u + b_u)) * sigma2_bar_u  # vector length N

        # ---- allocate arrays with burn-in ----
        T_total = self.T + self.burn_in
        K = self.n_factors
        N = self.n_firms

        f = np.zeros((T_total, K))
        sigma2_f = np.zeros((T_total, K))
        u = np.zeros((T_total, N))
        sigma2_u = np.zeros((T_total, N))

        # init variances at unconditional levels
        sigma2_f[0] = np.maximum(sigma2_bar_f, self.eps)
        sigma2_u[0] = np.maximum(sigma2_bar_u, self.eps)

        mu_f_vec = np.full(K, float(self.mu_f))

        # cross-sectional parameters
        alpha_i = self.rng.normal(0.0, self.alpha_i_std, size=N)
        beta_i = self.rng.normal(self.beta_mean, self.beta_std, size=(N, K))

        # ---- simulate ----
        for t in range(T_total):
            if t > 0:
                # realized shocks (previous)
                f_lag = f[t - 1] - mu_f_vec
                u_lag = u[t - 1]

                sigma2_f[t] = omega_f + a_f * (f_lag ** 2) + b_f * sigma2_f[t - 1]
                sigma2_u[t] = omega_u + a_u * (u_lag ** 2) + b_u * sigma2_u[t - 1]

                # numerical floor
                sigma2_f[t] = np.maximum(sigma2_f[t], self.eps)
                sigma2_u[t] = np.maximum(sigma2_u[t], self.eps)

            # draw standardized shocks
            z_f = self.rng.normal(size=K)
            z_u = self.rng.normal(size=N)

            # realize factor and idiosyncratic components
            f[t] = mu_f_vec + np.sqrt(sigma2_f[t]) * z_f
            u[t] = np.sqrt(sigma2_u[t]) * z_u

        # discard burn-in
        f_out = f[self.burn_in :]
        u_out = u[self.burn_in :]
        sigma2_f_out = sigma2_f[self.burn_in :]
        sigma2_u_out = sigma2_u[self.burn_in :]

        # returns
        r_out = alpha_i[None, :] + f_out @ beta_i.T + u_out

        self._simulation_result = {
            "returns": r_out,
            "factors": f_out,
            "sigma2_factors": sigma2_f_out,
            "sigma2_idio": sigma2_u_out,
            "firm_params": {
                "alpha_i": alpha_i,
                "beta_i": beta_i,
                "sigma2_bar_idio_i": sigma2_bar_u,
            },
            "params": {
                "case": "GARCH",
                "T": self.T,
                "burn_in": self.burn_in,
                "n_firms": self.n_firms,
                "n_factors": self.n_factors,
                "mu_f": self.mu_f,
                "rho_f": self.rho_f,
                "rho_u": self.rho_u,
                "alpha_share_f": self.alpha_share_f,
                "alpha_share_u": self.alpha_share_u,
                "a_f": float(a_f),
                "b_f": float(b_f),
                "a_u": float(a_u),
                "b_u": float(b_u),
                "sigma2_bar_factor": float(self.sigma2_bar_factor),
                "sigma2_bar_idio": float(self.sigma2_bar_idio),
                "idio_sigma_log": float(self.idio_sigma_log),
                "alpha_i_std": float(self.alpha_i_std),
                "beta_mean": float(self.beta_mean),
                "beta_std": float(self.beta_std),
                "eps": float(self.eps),
                "seed": self.seed,
            },
        }
        return self._simulation_result

    def _convert_to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Convert simulation results to a tidy long DataFrame."""
        returns = result["returns"]
        sigma2_idio = result["sigma2_idio"]
        T, N = returns.shape

        df_list = []
        for i in range(N):
            df_firm = pd.DataFrame(
                {
                    "time": np.arange(T),
                    "series_id": i,
                    "y": returns[:, i],
                    "sigma2_idio": sigma2_idio[:, i],
                }
            )
            df_list.append(df_firm)

        df = pd.concat(df_list, ignore_index=True)
        return df
