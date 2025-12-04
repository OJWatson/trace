"""
TRACE: Temporal and Regional Analysis of Conflict Events

A Bayesian hierarchical modeling package for conflict casualty analysis.
"""

__version__ = "0.1.0"
__author__ = "Imperial College London"
__email__ = "trace@imperial.ac.uk"

# Import submodules
from . import data, model, simulate, analysis

__all__ = ["data", "model", "simulate", "analysis", "__version__"]
