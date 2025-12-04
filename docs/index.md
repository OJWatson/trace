# TRACE Documentation

**TRACE** (Temporal and Regional Analysis of Conflict Events) is a Python package for analyzing conflict casualties using semi-mechanistic Bayesian hierarchical models.

## Overview

TRACE integrates multiple data streams—conflict events from ACLED, hospital admissions, and national mortality data—to provide insights into casualty dynamics and enable short-term forecasting. The package is built on JAX and NumPyro for fast, GPU-accelerated Bayesian inference.

## Key Features

- **Semi-Mechanistic Bayesian Models**: Combines mechanistic understanding with data-driven inference
- **Spatial Modeling**: Distance-based kernels for distributing casualties among hospitals
- **Temporal Dynamics**: Delay distributions from injury to death
- **Multiple Data Streams**: Integrates conflict events, hospital admissions, and mortality
- **Fast Inference**: JAX/NumPyro backend with GPU support
- **Forecasting**: Project future casualties under different scenarios
- **Validation Tools**: Synthetic data generation for model testing

## Quick Start

```python
import trace
from trace.data import fetch_acled_data, prepare_acled_events
from trace.analysis import run_inference, plot_fit

# Fetch conflict event data
events_df = fetch_acled_data(
    country="Palestine",
    start_date="2023-01-01",
    end_date="2023-12-31",
    api_token="YOUR_TOKEN"
)

# Prepare for modeling
events_by_day, event_days, event_coords, dates = prepare_acled_events(
    events_df, "2023-01-01", "2023-12-31"
)

# Run inference
mcmc, samples = run_inference(
    events_by_day=events_by_day,
    event_day_index=event_days,
    event_coords=event_coords,
    hospital_coords=hospital_locations,
    injuries_obs=hospital_data,
    deaths_obs=mortality_data
)

# Visualize results
plot_fit(dates, injuries_obs, deaths_obs, posterior_predictive)
```

## Installation

### From PyPI

```bash
pip install trace-conflict
```

### From Source

```bash
git clone https://github.com/imperial-college/trace.git
cd trace
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev,docs]"
```

## Contents

```{toctree}
:maxdepth: 2
:caption: Model Documentation

model/00_overview
model/01_background
model/02_description
model/03_implementation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/01_basic_example.ipynb
tutorials/02_gaza_analysis
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/data
api/model
api/simulate
api/analysis
```

```{toctree}
:maxdepth: 1
:caption: Additional Information

contributing
```

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
