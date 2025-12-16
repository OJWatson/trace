"""
TRACE: Temporal and Regional Analysis of Conflict Events

A Bayesian hierarchical modeling package for conflict casualty analysis.
"""

import importlib

__version__ = "0.1.0"
__author__ = "Imperial College London"
__email__ = "trace@imperial.ac.uk"

# Import submodules

__all__ = ["data", "model", "simulate", "analysis", "__version__"]


def __getattr__(name: str):
    if name in {"analysis", "data", "model", "simulate"}:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + ["analysis", "data", "model", "simulate"])
