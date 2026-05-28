"""Compatibility shim for older pip editable installs."""

from setuptools import find_packages, setup


setup(
    name="finstressts-paper-code",
    version="0.1.0",
    description="Paper-code release for FinStressTS probabilistic financial stress time-series forecasting experiments.",
    packages=find_packages(include=["finprobts", "finprobts.*"]),
    python_requires=">=3.9",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest"],
        "parquet": ["pyarrow"],
        "torch": ["torch"],
        "deep": ["torch"],
    },
    entry_points={
        "console_scripts": [
            "finprobts=finprobts.cli:main",
        ]
    },
)
