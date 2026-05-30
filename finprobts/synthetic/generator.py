"""Synthetic benchmark dataset generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from finprobts.synthetic.presets import CASE_PRESETS, get_case_config, list_cases


ALL_CASES = list_cases()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        item = value.item()
        return None if isinstance(item, float) and not math.isfinite(item) else item
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _har_coeffs_from_s_lambda(s: float, lam: float) -> tuple[float, float, float]:
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}.")
    if not 0.0 <= s < 1.0:
        raise ValueError(f"s must be in [0, 1), got {s}.")
    b22 = lam * s
    rem = (1.0 - lam) * s
    return float(rem / 2.0), float(rem / 2.0), float(b22)


def _normalize_tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    if "firm_id" in df.columns and "series_id" not in df.columns:
        rename_map["firm_id"] = "series_id"
    if "return" in df.columns and "y" not in df.columns:
        rename_map["return"] = "y"
    if rename_map:
        df = df.rename(columns=rename_map)
    required = {"time", "series_id", "y"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Synthetic dataframe missing required columns: {sorted(missing)}")
    return df


def _summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_series": int(df["series_id"].nunique()) if "series_id" in df.columns else None,
        "T_effective": int(df["time"].nunique()) if "time" in df.columns else None,
        "mean_y": float(df["y"].mean()) if "y" in df.columns else None,
        "std_y": float(df["y"].std()) if "y" in df.columns else None,
    }
    if {"series_id", "time", "y"}.issubset(df.columns):
        d = df.sort_values(["series_id", "time"]).copy()
        d["abs_y"] = d["y"].abs()
        d["abs_y_lag1"] = d.groupby("series_id")["abs_y"].shift(1)
        corrs = []
        for _, group in d.dropna(subset=["abs_y_lag1"]).groupby("series_id"):
            corrs.append(_safe_corr(group["abs_y"].to_numpy(), group["abs_y_lag1"].to_numpy()))
        summary["vol_clustering_corr_abs_lag1_mean"] = float(np.nanmean(corrs)) if corrs else None
    for column in ["sigma2_idio", "event_count", "intensity", "jump_size", "is_outlier", "regime"]:
        if column in df.columns:
            values = df[column]
            if values.dtype == bool:
                summary[f"{column}_rate"] = float(values.mean())
            elif np.issubdtype(values.dtype, np.number):
                summary[f"{column}_mean"] = float(values.mean())
    return summary


def _make_simulator(case: str, cfg: Dict[str, Any], seed: int):
    if case == "case1_garch":
        from finprobts.simulators.garch import GARCHSimulator

        return GARCHSimulator(
            T=cfg["T"], n_firms=cfg["n_firms"], n_factors=cfg["n_factors"],
            rho_f=cfg["rho_f"], rho_u=cfg["rho_u"],
            alpha_share_f=cfg["alpha_share_f"], alpha_share_u=cfg["alpha_share_u"],
            sigma2_bar_factor=cfg["sigma2_bar_factor"], sigma2_bar_idio=cfg["sigma2_bar_idio"],
            idio_sigma_log=cfg["idio_sigma_log"], mu_f=cfg["mu_f"],
            alpha_i_std=cfg["alpha_i_std"], beta_mean=cfg["beta_mean"], beta_std=cfg["beta_std"],
            burn_in=cfg["burn_in"], eps=cfg["eps"], seed=seed,
        )
    if case == "case2_har":
        from finprobts.simulators.har import HARSimulator

        b1, b5, b22 = _har_coeffs_from_s_lambda(cfg["s"], cfg["lam"])
        cfg["b1"] = b1
        cfg["b5"] = b5
        cfg["b22"] = b22
        cfg["c_factor"] = cfg["gamma"] * cfg["c_idio"]
        return HARSimulator(
            T=cfg["T"], n_firms=cfg["n_firms"], n_factors=cfg["n_factors"],
            b1_u=b1, b5_u=b5, b22_u=b22, c_idio=cfg["c_idio"],
            b1_f=b1, b5_f=b5, b22_f=b22, c_factor=cfg["c_factor"],
            burn_in=cfg["burn_in"], eps=cfg["eps"], seed=seed,
        )
    if case == "case3_heavy_tail":
        from finprobts.simulators.heavy_tail import HeavyTailSimulator

        return HeavyTailSimulator(
            T=cfg["T"], n_firms=cfg["n_firms"], n_factors=cfg["n_factors"],
            rho_v=cfg["rho_v"], sigma2_bar_factor=cfg["sigma2_bar_factor"],
            sigma2_bar_idio=cfg["sigma2_bar_idio"], nu=cfg["nu"],
            pi_outlier=cfg["pi_outlier"], outlier_scale=cfg["outlier_scale"],
            burn_in=cfg["burn_in"], seed=seed,
        )
    if case == "case4_regime":
        from finprobts.simulators.regime_switching import MarketRegimePanelSimulator

        return MarketRegimePanelSimulator(
            T=cfg["T"], n_firms=cfg["n_firms"], block_size=cfg["block_size"],
            mu_U=cfg["mu_U"], mu_S=cfg["mu_S"], mu_D=cfg["mu_D"],
            sigma_U=cfg["sigma_U"], sigma_S=cfg["sigma_S"], sigma_D=cfg["sigma_D"],
            phi=cfg["phi"], Pi_block=cfg["Pi_block"],
            mu_scale_logsigma=cfg["mu_scale_logsigma"], sig_scale_logsigma=cfg["sig_scale_logsigma"],
            burn_in=cfg["burn_in"], seed=seed,
        )
    if case == "case5_hawkes":
        from finprobts.simulators.hawkes import MarketHawkesPanelSimulator

        return MarketHawkesPanelSimulator(
            T=cfg["T"], burn_in=cfg["burn_in"], n_firms=cfg["n_firms"],
            n_factors=cfg["n_factors"], phi=cfg["phi"], sigma_eps=cfg["sigma_eps"],
            alpha_i_std=cfg["alpha_i_std"], mu=cfg["mu"], alpha=cfg["alpha"],
            beta=cfg["beta"], jump_mean_abs=cfg["jump_mean_abs"],
            jump_sigma_log=cfg["jump_sigma_log"], p_up=cfg["p_up"],
            gamma_mean=cfg["gamma_mean"], gamma_logsigma=cfg["gamma_logsigma"],
            eps=cfg["eps"], seed=seed,
        )
    if case == "case6_zip_panel":
        from finprobts.simulators.zero_inflated import MarketZIPPanelSimulator

        return MarketZIPPanelSimulator(
            T=cfg["T"], n_firms=cfg["n_firms"], n_factors=cfg["n_factors"],
            phi=cfg["phi"], sigma_eps=cfg["sigma_eps"], pi=cfg["pi"], lam=cfg["lam"],
            jump_mean_abs=cfg["jump_mean_abs"], jump_sigma_log=cfg["jump_sigma_log"],
            p_up=cfg["p_up"], gamma_mean=cfg["gamma_mean"], gamma_std=cfg["gamma_std"],
            alpha_i_std=cfg["alpha_i_std"], beta_mean=cfg["beta_mean"], beta_std=cfg["beta_std"],
            mu_f=cfg["mu_f"], sigma_f=cfg["sigma_f"], burn_in=cfg["burn_in"], seed=seed,
        )
    raise KeyError(f"Unknown synthetic case '{case}'.")


def _tag(case: str, cfg: Dict[str, Any], seed: int) -> str:
    return f"{case}_level{int(cfg['level']):02d}_N{int(cfg.get('n_firms', 1))}_T{int(cfg['T'])}_seed{seed}"


def _write_artifacts(
    df: pd.DataFrame,
    out_dir: Path,
    tag: str,
    formats: Sequence[str],
    meta: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    paths: Dict[str, Optional[str]] = {"csv_path": None, "parquet_path": None}
    if "csv" in formats:
        csv_path = out_dir / f"{tag}.csv"
        df.to_csv(csv_path, index=False)
        paths["csv_path"] = str(csv_path)
    if "parquet" in formats:
        parquet_path = out_dir / f"{tag}.parquet"
        df.to_parquet(parquet_path, index=False)
        paths["parquet_path"] = str(parquet_path)
    meta_path = out_dir / f"{tag}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(meta), handle, indent=2, allow_nan=False)
    paths["meta_path"] = str(meta_path)
    return paths


def _parse_formats(formats: Iterable[str]) -> List[str]:
    normalized = []
    for item in formats:
        for fmt in str(item).split(","):
            fmt = fmt.strip().lower()
            if fmt:
                normalized.append(fmt)
    invalid = sorted(set(normalized) - {"csv", "parquet"})
    if invalid:
        raise ValueError(f"Unsupported output formats: {invalid}. Use csv and/or parquet.")
    return sorted(set(normalized)) or ["csv"]


def generate_synthetic_case(
    case: str,
    level: int,
    out_dir: str = "data/simulated",
    base_seed: Optional[int] = None,
    T: Optional[int] = None,
    n_firms: Optional[int] = None,
    formats: Iterable[str] = ("csv",),
) -> Dict[str, Any]:
    """Generate one notebook-style synthetic benchmark dataset."""

    case = case.lower()
    cfg = get_case_config(case, level)
    if T is not None:
        cfg["T"] = int(T)
    if n_firms is not None and "n_firms" in cfg:
        cfg["n_firms"] = int(n_firms)
    seed_base = CASE_PRESETS[case]["default_base_seed"] if base_seed is None else int(base_seed)
    seed = seed_base + int(level)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sim = _make_simulator(case, cfg, seed)
    result = sim.simulate()
    df = _normalize_tidy_columns(sim.to_dataframe())
    df["case"] = case
    df["difficulty_level"] = int(level)

    tag = _tag(case, cfg, seed)
    summary = _summary(df)
    meta = {
        "case": case,
        "title": CASE_PRESETS[case]["title"],
        "tag": tag,
        "level": int(level),
        "seed": seed,
        "config": cfg,
        "sim_params_dump": result.get("params", {}),
        "summary": summary,
    }
    paths = _write_artifacts(df, out_path, tag, _parse_formats(formats), meta)
    return {
        "case": case,
        "level": int(level),
        "tag": tag,
        "seed": seed,
        "config": _json_safe(cfg),
        "summary": _json_safe(summary),
        **paths,
    }


def _parse_cases(case: str) -> List[str]:
    if case.lower() == "all":
        return ALL_CASES
    cases = [part.strip().lower() for part in case.split(",") if part.strip()]
    unknown = sorted(set(cases) - set(ALL_CASES))
    if unknown:
        raise KeyError(f"Unknown synthetic cases: {unknown}. Available cases: {', '.join(ALL_CASES)}")
    return cases


def generate_synthetic_suite(
    case: str = "all",
    levels: Iterable[int] = (1, 2, 3, 4, 5),
    out_dir: str = "data/simulated",
    base_seed: Optional[int] = None,
    T: Optional[int] = None,
    n_firms: Optional[int] = None,
    formats: Iterable[str] = ("csv",),
) -> Dict[str, Any]:
    """Generate a suite of synthetic datasets and write a manifest."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cases = _parse_cases(case)
    level_list = [int(level) for level in levels]
    outputs = []
    for case_name in cases:
        for level in level_list:
            outputs.append(
                generate_synthetic_case(
                    case=case_name,
                    level=level,
                    out_dir=str(out_path),
                    base_seed=base_seed,
                    T=T,
                    n_firms=n_firms,
                    formats=formats,
                )
            )
    manifest = {
        "cases": cases,
        "levels": level_list,
        "out_dir": str(out_path),
        "datasets": outputs,
    }
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, indent=2, allow_nan=False)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
