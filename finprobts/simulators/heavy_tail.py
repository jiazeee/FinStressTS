from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base import BaseSimulator


class HeavyTailSimulator(BaseSimulator):
    """
    Case 3: Heavy-tailed innovations + rare contamination outliers.

    r_{i,t} = alpha_i + beta_i' f_t + u_{i,t}
    f_{j,t} = mu_f + sqrt(sigma^2_{f,j,t}) * z_{f,j,t}
    u_{i,t} = u_base_{i,t} + o_{i,t}
    u_base_{i,t} = sqrt(sigma^2_{u,i,t}) * z_{u,i,t}

    - z are standardized Student-t (unit variance).
    - o is a rare additive outlier shock that does NOT feed back into volatility.
    """

    def __init__(
        self,
        T: int = 2000,
        n_firms: int = 50,
        n_factors: int = 3,
        rho_v: float = 0.9,
        sigma2_bar_factor: float = 0.02,
        sigma2_bar_idio: float = 0.04,
        nu: float = 5.0,
        pi_outlier: float = 0.0,
        outlier_scale: float = 10.0,
        burn_in: int = 200,
        seed: Optional[int] = None,
        alpha_i_std: float = 0.001,
        beta_mean: float = 0.5,
        beta_std: float = 0.2,
        mu_f: float = 0.0,
        alpha_share: float = 0.10,
        eps: float = 1e-12,
    ):
        super().__init__(seed)

        if nu <= 2.0:
            raise ValueError("nu must be > 2 for finite variance Student-t.")
        if not (0.0 <= pi_outlier <= 1.0):
            raise ValueError("pi_outlier must be in [0, 1].")
        if not (0.0 < rho_v < 1.0):
            raise ValueError("rho_v must be in (0, 1) for stationarity.")
        if sigma2_bar_factor <= 0 or sigma2_bar_idio <= 0:
            raise ValueError("Unconditional variances must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0.")
        if not (0.0 < alpha_share < 1.0):
            raise ValueError("alpha_share must be in (0, 1).")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.T = int(T)
        self.n_firms = int(n_firms)
        self.n_factors = int(n_factors)

        self.rho_v = float(rho_v)
        self.sigma2_bar_factor = float(sigma2_bar_factor)
        self.sigma2_bar_idio = float(sigma2_bar_idio)

        self.nu = float(nu)
        self.pi_outlier = float(pi_outlier)
        self.outlier_scale = float(outlier_scale)

        self.burn_in = int(burn_in)

        self.alpha_i_std = float(alpha_i_std)
        self.beta_mean = float(beta_mean)
        self.beta_std = float(beta_std)
        self.mu_f = float(mu_f)

        self.alpha_share = float(alpha_share)
        self.eps = float(eps)

    def _draw_student_t_standard(self, size: tuple[int, ...]) -> np.ndarray:
        """Standardized Student-t with unit variance."""
        z_raw = self.rng.standard_t(df=self.nu, size=size)
        scale = np.sqrt((self.nu - 2.0) / self.nu)
        return scale * z_raw

    def simulate(self) -> Dict[str, Any]:
        T_full = self.T + self.burn_in
        K, N = self.n_factors, self.n_firms

        # GARCH params
        a = self.alpha_share * self.rho_v
        b = self.rho_v - a

        omega_f = (1.0 - self.rho_v) * self.sigma2_bar_factor
        omega_u = (1.0 - self.rho_v) * self.sigma2_bar_idio

        # Allocate
        f = np.zeros((T_full, K), dtype=float)
        u_base = np.zeros((T_full, N), dtype=float)   # excludes contamination
        u = np.zeros((T_full, N), dtype=float)        # includes contamination

        sigma2_f = np.full((T_full, K), self.sigma2_bar_factor, dtype=float)
        sigma2_u = np.full((T_full, N), self.sigma2_bar_idio, dtype=float)

        z_f = np.zeros((T_full, K), dtype=float)
        z_u = np.zeros((T_full, N), dtype=float)

        outlier_mask = np.zeros((T_full, N), dtype=bool)

        # Cross-sectional params
        alpha_i = self.rng.normal(0.0, self.alpha_i_std, size=N)
        beta_i = self.rng.normal(self.beta_mean, self.beta_std, size=(N, K))
        mu_f_vec = np.full(K, self.mu_f, dtype=float)

        for t in range(T_full):
            if t > 0:
                # realized innovations for variance recursion
                # factors: (f_{t-1} - mu_f)^2
                f_innov_sq = (f[t - 1] - mu_f_vec) ** 2

                # idio: use u_base only (so outliers do NOT feed back)
                u_innov_sq = u_base[t - 1] ** 2

                sigma2_f[t] = omega_f + a * f_innov_sq + b * sigma2_f[t - 1]
                sigma2_u[t] = omega_u + a * u_innov_sq + b * sigma2_u[t - 1]

                sigma2_f[t] = np.maximum(sigma2_f[t], self.eps)
                sigma2_u[t] = np.maximum(sigma2_u[t], self.eps)

            # draw shocks
            z_f[t] = self._draw_student_t_standard((K,))
            z_u[t] = self._draw_student_t_standard((N,))

            # realize base components
            f[t] = mu_f_vec + np.sqrt(sigma2_f[t]) * z_f[t]
            u_base[t] = np.sqrt(sigma2_u[t]) * z_u[t]

            # contaminate (additive; no feedback because recursion uses u_base)
            u[t] = u_base[t]
            if self.pi_outlier > 0.0:
                mask = self.rng.random(N) < self.pi_outlier
                if np.any(mask):
                    signs = np.sign(self.rng.normal(size=int(mask.sum())))
                    u[t, mask] += self.outlier_scale * np.sqrt(sigma2_u[t, mask]) * signs
                    outlier_mask[t, mask] = True

        # drop burn-in
        sl = slice(self.burn_in, T_full)
        f = f[sl]
        u = u[sl]
        sigma2_f = sigma2_f[sl]
        sigma2_u = sigma2_u[sl]
        outlier_mask = outlier_mask[sl]

        r = alpha_i[None, :] + f @ beta_i.T + u

        self._simulation_result = {
            "returns": r,
            "factors": f,
            "sigma2_factors": sigma2_f,
            "sigma2_idio": sigma2_u,
            "outlier_mask_idio": outlier_mask,
            "params": {
                "case": "HEAVY_TAILS_OUTLIERS",
                "T": self.T,
                "burn_in": self.burn_in,
                "n_firms": N,
                "n_factors": K,
                "rho_v": self.rho_v,
                "alpha": float(a),
                "beta": float(b),
                "sigma2_bar_factor": self.sigma2_bar_factor,
                "sigma2_bar_idio": self.sigma2_bar_idio,
                "nu": self.nu,
                "pi_outlier": self.pi_outlier,
                "outlier_scale": self.outlier_scale,
                "alpha_i_std": self.alpha_i_std,
                "beta_mean": self.beta_mean,
                "beta_std": self.beta_std,
                "mu_f": self.mu_f,
                "eps": self.eps,
                "seed": self.seed,
            },
        }
        return self._simulation_result

    def _convert_to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        r = result["returns"]
        sigma2_u = result["sigma2_idio"]
        outlier = result["outlier_mask_idio"]
        T, N = r.shape

        dfs = []
        for i in range(N):
            dfs.append(
                pd.DataFrame(
                    {
                        "time": np.arange(T),
                        "series_id": i,
                        "y": r[:, i],
                        "sigma2_idio": sigma2_u[:, i],
                        "is_outlier": outlier[:, i],
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)
