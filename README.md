# TRACE: Conflict Casualty Modeling with Bayesian Hierarchical Models

**TRACE** (Temporal and Regional Analysis of Conflict Events) is a Python package for analyzing conflict casualties using semi-mechanistic Bayesian hierarchical models. Inspired by epidemiological modeling approaches (Flaxman et al., 2020), TRACE integrates multiple data streams—conflict events, hospital admissions, and mortality data—to provide insights into casualty dynamics and enable short-term forecasting.

## Features

- **Semi-Mechanistic Bayesian Models**: Combines mechanistic understanding with data-driven inference
- **Spatial Modeling**: Distance-based kernels for distributing casualties among hospitals
- **Temporal Dynamics**: Delay distributions from injury to death
- **Multiple Data Streams**: Integrates ACLED conflict events, hospital admissions, and national mortality
- **JAX/NumPyro Backend**: Fast, GPU-accelerated Bayesian inference with MCMC
- **Forecasting**: Project future casualties under different scenarios
- **Synthetic Data Generation**: Validate models with simulated data

## Installation

### From PyPI (when released)

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

## Quick Start

```python
import trace
from trace.data import fetch_acled_data, prepare_acled_events
from trace.model import casualty_model
from trace.analysis import run_inference, plot_fit

# Fetch conflict event data
events_df = fetch_acled_data(
    country="Palestine",
    start_date="2023-01-01",
    end_date="2023-12-31",
    api_token="YOUR_ACLED_TOKEN"
)

# Prepare data for modeling
events_by_day, event_days, event_coords, dates = prepare_acled_events(
    events_df, "2023-01-01", "2023-12-31"
)

# Run Bayesian inference
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

## Model Overview

TRACE implements a hierarchical Bayesian model that:

1. **Models conflict intensity**: Treats conflict events as generating casualties through a latent intensity process
2. **Spatial allocation**: Distributes injured individuals to hospitals based on distance (exponential decay kernel)
3. **Delay dynamics**: Models the delay from injury to death using discrete probability distributions
4. **Joint likelihood**: Combines hospital admissions and mortality data in a coherent probabilistic framework

### Key Parameters

- `mu_w`: Average number of wounded per conflict event
- `mu_i`: Average number of immediate fatalities per event
- `p_late`: Probability an injured person eventually dies (hospital fatality rate)
- `ell`: Spatial length scale for hospital allocation
- `delay_probs`: Distribution of delays from injury to death

## Documentation

Full documentation is available at [https://ojwatson.github.io/trace](https://ojwatson.github.io/trace)

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:

- Basic usage with synthetic data
- Real-world analysis with ACLED data
- Forecasting under different scenarios
- Model validation and diagnostics

## Citation

If you use TRACE in your research, please cite:

```bibtex
@software{trace2024,
  title = {TRACE: Conflict Casualty Modeling with Bayesian Hierarchical Models},
  author = {{OJWatson}},
  year = {2024},
  url = {https://github.com/OJWatson/trace}
}
```

## Related Work

This package draws inspiration from:

- Flaxman et al. (2020) - COVID-19 modeling with semi-mechanistic approaches
- The `epidemia` R package for epidemic modeling
- Chemaitelly et al. (2024) - Conflict mortality as hazard rates
- LSHTM-JH Gaza projections - Scenario-based health impact modeling

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This work was developed at Imperial College London as part of research into conflict casualty dynamics and humanitarian forecasting.
