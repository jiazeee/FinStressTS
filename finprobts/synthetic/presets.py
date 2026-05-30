"""Notebook-derived synthetic dataset presets.

The presets here intentionally expose a compact benchmark surface: six
financial stylized-fact cases, five diagnostic levels each, and fixed defaults
for the less commonly tuned simulator parameters.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


CASE_PRESETS: Dict[str, Dict[str, Any]] = {
    "case1_garch": {
        "title": "Volatility clustering with GARCH factor/idiosyncratic dynamics",
        "default_base_seed": 123,
        "fixed": {
            "T": 20000,
            "n_factors": 3,
            "n_firms": 50,
            "sigma2_bar_idio": 0.04,
            "alpha_share_f": 0.10,
            "alpha_share_u": 0.10,
            "mu_f": 0.0,
            "alpha_i_std": 0.001,
            "beta_mean": 0.5,
            "beta_std": 0.2,
            "burn_in": 200,
            "eps": 1e-10,
        },
        "levels": [
            {"level": 1, "name": "baseline_balanced", "rho_f": 0.80, "rho_u": 0.80, "idio_sigma_log": 0.10, "sigma2_bar_factor": 0.02},
            {"level": 2, "name": "factor_persistent", "rho_f": 0.95, "rho_u": 0.75, "idio_sigma_log": 0.10, "sigma2_bar_factor": 0.03},
            {"level": 3, "name": "idio_persistent", "rho_f": 0.75, "rho_u": 0.95, "idio_sigma_log": 0.10, "sigma2_bar_factor": 0.015},
            {"level": 4, "name": "high_heterogeneity", "rho_f": 0.80, "rho_u": 0.80, "idio_sigma_log": 0.60, "sigma2_bar_factor": 0.02},
            {"level": 5, "name": "low_common_snr", "rho_f": 0.85, "rho_u": 0.85, "idio_sigma_log": 0.10, "sigma2_bar_factor": 0.005},
        ],
    },
    "case2_har": {
        "title": "HAR multi-scale volatility memory",
        "default_base_seed": 123,
        "fixed": {
            "T": 20000,
            "n_factors": 3,
            "burn_in": 200,
            "eps": 1e-12,
        },
        "levels": [
            {"level": 1, "s": 0.60, "lam": 0.40, "c_idio": 2e-4, "gamma": 1.00, "n_firms": 50},
            {"level": 2, "s": 0.90, "lam": 0.40, "c_idio": 2e-4, "gamma": 1.00, "n_firms": 50},
            {"level": 3, "s": 0.60, "lam": 0.70, "c_idio": 2e-4, "gamma": 1.00, "n_firms": 50},
            {"level": 4, "s": 0.60, "lam": 0.40, "c_idio": 1e-3, "gamma": 1.00, "n_firms": 50},
            {"level": 5, "s": 0.60, "lam": 0.40, "c_idio": 2e-4, "gamma": 4.00, "n_firms": 50},
        ],
    },
    "case3_heavy_tail": {
        "title": "Heavy tails and rare outlier contamination",
        "default_base_seed": 123,
        "fixed": {
            "T": 20000,
            "n_factors": 3,
            "sigma2_bar_factor": 0.02,
            "sigma2_bar_idio": 0.04,
            "burn_in": 200,
        },
        "levels": [
            {"level": 1, "nu": 8, "pi_outlier": 0.00, "outlier_scale": 6, "rho_v": 0.90, "n_firms": 50},
            {"level": 2, "nu": 3, "pi_outlier": 0.00, "outlier_scale": 6, "rho_v": 0.90, "n_firms": 50},
            {"level": 3, "nu": 8, "pi_outlier": 0.02, "outlier_scale": 6, "rho_v": 0.90, "n_firms": 50},
            {"level": 4, "nu": 8, "pi_outlier": 0.005, "outlier_scale": 12, "rho_v": 0.90, "n_firms": 50},
            {"level": 5, "nu": 3, "pi_outlier": 0.02, "outlier_scale": 12, "rho_v": 0.90, "n_firms": 50},
        ],
    },
    "case4_regime": {
        "title": "Market-wide block Markov regime switching",
        "default_base_seed": 2025,
        "fixed": {
            "T": 20000,
            "n_firms": 50,
            "Pi_block": None,
            "burn_in": 200,
            "mu_scale_logsigma": 0.10,
            "sig_scale_logsigma": 0.10,
        },
        "levels": [
            {"level": 1, "block_size": 50, "mu_U": 0.0012, "mu_S": 0.0, "mu_D": -0.0012, "sigma_U": 0.010, "sigma_S": 0.0085, "sigma_D": 0.016, "phi": 0.20},
            {"level": 2, "block_size": 10, "mu_U": 0.0012, "mu_S": 0.0, "mu_D": -0.0012, "sigma_U": 0.010, "sigma_S": 0.0085, "sigma_D": 0.016, "phi": 0.20},
            {"level": 3, "block_size": 50, "mu_U": 0.0005, "mu_S": 0.0, "mu_D": -0.0005, "sigma_U": 0.010, "sigma_S": 0.0095, "sigma_D": 0.011, "phi": 0.20},
            {"level": 4, "block_size": 50, "mu_U": 0.0020, "mu_S": 0.0, "mu_D": -0.0020, "sigma_U": 0.008, "sigma_S": 0.0075, "sigma_D": 0.020, "phi": 0.20},
            {"level": 5, "block_size": 50, "mu_U": 0.0012, "mu_S": 0.0, "mu_D": -0.0012, "sigma_U": 0.010, "sigma_S": 0.0085, "sigma_D": 0.016, "phi": 0.60},
        ],
    },
    "case5_hawkes": {
        "title": "Market-wide self-exciting jumps",
        "default_base_seed": 2025,
        "fixed": {
            "T": 20000,
            "burn_in": 200,
            "n_firms": 50,
            "n_factors": 0,
            "phi": 0.1,
            "sigma_eps": 0.01,
            "alpha_i_std": 0.001,
            "p_up": 0.5,
            "gamma_mean": 1.0,
            "gamma_logsigma": 0.0,
            "eps": 1e-8,
        },
        "levels": [
            {"level": 1, "alpha": 0.06, "beta": 1.2, "mu": 0.05, "jump_mean_abs": 0.02, "jump_sigma_log": 0.45},
            {"level": 2, "alpha": 0.18, "beta": 1.2, "mu": 0.05, "jump_mean_abs": 0.02, "jump_sigma_log": 0.45},
            {"level": 3, "alpha": 0.25, "beta": 0.4, "mu": 0.05, "jump_mean_abs": 0.02, "jump_sigma_log": 0.45},
            {"level": 4, "alpha": 0.06, "beta": 1.2, "mu": 0.15, "jump_mean_abs": 0.02, "jump_sigma_log": 0.45},
            {"level": 5, "alpha": 0.06, "beta": 1.2, "mu": 0.05, "jump_mean_abs": 0.05, "jump_sigma_log": 1.00},
        ],
    },
    "case6_zip_panel": {
        "title": "Market-wide zero-inflated Poisson jumps with panel exposure",
        "default_base_seed": 2025,
        "fixed": {
            "T": 20000,
            "burn_in": 200,
            "n_firms": 50,
            "n_factors": 0,
            "sigma_eps": 0.01,
            "mu_f": 0.0,
            "sigma_f": 0.01,
            "p_up": 0.5,
            "gamma_mean": 1.0,
            "gamma_std": 0.2,
            "alpha_i_std": 0.001,
            "beta_mean": 0.5,
            "beta_std": 0.2,
        },
        "levels": [
            {"level": 1, "pi": 0.70, "lam": 0.20, "jump_mean_abs": 0.030, "jump_sigma_log": 0.60, "phi": 0.20},
            {"level": 2, "pi": 0.90, "lam": 0.20, "jump_mean_abs": 0.030, "jump_sigma_log": 0.60, "phi": 0.20},
            {"level": 3, "pi": 0.70, "lam": 0.60, "jump_mean_abs": 0.030, "jump_sigma_log": 0.60, "phi": 0.20},
            {"level": 4, "pi": 0.70, "lam": 0.20, "jump_mean_abs": 0.080, "jump_sigma_log": 1.00, "phi": 0.20},
            {"level": 5, "pi": 0.70, "lam": 0.20, "jump_mean_abs": 0.030, "jump_sigma_log": 0.60, "phi": 0.55},
        ],
    },
}


def list_cases() -> List[str]:
    """Return supported synthetic benchmark case names."""

    return sorted(CASE_PRESETS)


def get_case_config(case: str, level: int) -> Dict[str, Any]:
    """Return merged fixed and level config for a case/level pair."""

    if case not in CASE_PRESETS:
        available = ", ".join(list_cases())
        raise KeyError(f"Unknown synthetic case '{case}'. Available cases: {available}")
    preset = CASE_PRESETS[case]
    for level_cfg in preset["levels"]:
        if int(level_cfg["level"]) == int(level):
            cfg = deepcopy(preset["fixed"])
            cfg.update(deepcopy(level_cfg))
            return cfg
    raise KeyError(f"Case '{case}' does not define level {level}.")
