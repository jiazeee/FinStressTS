"""Synthetic financial dataset generation."""

from finprobts.synthetic.generator import generate_synthetic_case, generate_synthetic_suite
from finprobts.synthetic.presets import CASE_PRESETS, list_cases

__all__ = [
    "CASE_PRESETS",
    "generate_synthetic_case",
    "generate_synthetic_suite",
    "list_cases",
]
