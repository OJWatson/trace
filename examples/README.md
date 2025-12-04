# TRACE Examples

This directory contains examples demonstrating how to use TRACE for conflict casualty analysis.

## Examples

### 01_basic_usage.py

Basic introduction to TRACE covering:
- Simulating synthetic conflict data
- Fitting the Bayesian model
- Posterior predictive checks
- Generating forecasts
- Scenario analysis

**Run with:**
```bash
python examples/01_basic_usage.py
```

### Notebooks (Coming Soon)

- `02_acled_data_analysis.ipynb`: Working with real ACLED data
- `03_spatial_analysis.ipynb`: Deep dive into spatial modeling
- `04_forecasting_scenarios.ipynb`: Advanced forecasting techniques
- `05_model_validation.ipynb`: Comprehensive model validation

## Requirements

To run the examples, install TRACE with:

```bash
pip install trace-conflict
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Data

Some examples use synthetic data (generated automatically). For real-world examples:

1. **ACLED Data**: Register for API access at https://developer.acleddata.com/
2. **Hospital Data**: Format as CSV with columns: `date`, `hospital_id`, `count`
3. **Mortality Data**: Format as CSV with columns: `date`, `deaths`

## Output

Examples generate plots saved to this directory:
- `model_fit.png`: Observed vs fitted data
- `forecast.png`: Future projections
- Additional diagnostic plots

## Tips

- Start with `01_basic_usage.py` to understand the workflow
- Adjust MCMC parameters (`num_warmup`, `num_samples`) based on your computational resources
- Use smaller datasets for initial exploration
- Check convergence diagnostics (R-hat, effective sample size)

## Support

For questions or issues:
- Open an issue: https://github.com/imperial-college/trace/issues
- Email: trace@imperial.ac.uk
- Documentation: https://imperial-college.github.io/trace
