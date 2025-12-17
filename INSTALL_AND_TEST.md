# Installation and Testing Guide

This guide will help you install TRACE and verify that everything is working correctly.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/OJWatson/trace.git
cd trace
```

### 2. Create a Virtual Environment (Recommended)

**On Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the Package

**For users:**
```bash
pip install -e .
```

**For developers:**
```bash
pip install -e ".[dev,docs]"
```

### 4. Verify Installation

```bash
python -c "import trace; print(f'TRACE version {trace.__version__} installed successfully!')"
```

Expected output:
```
TRACE version 0.1.0 installed successfully!
```

## Quick Test

Run the basic example to verify everything works:

```bash
python examples/01_basic_usage.py
```

This will:
1. Generate synthetic conflict data
2. Fit the Bayesian model
3. Generate forecasts
4. Create visualizations

**Expected runtime**: 2-5 minutes (depending on your hardware)

**Expected output**: 
- Console output showing simulation and inference progress
- Two PNG files in `examples/`: `model_fit.png` and `forecast.png`

## Running Tests

Note: this repository includes a repo-local `pytest.ini` to ensure tests run reliably in a wide range of Python environments:

- It disables the external `pytest_httpserver` plugin (which can be incompatible with some `werkzeug` versions).
- It ensures `src/` is on the import path so `import trace` resolves to this package (and not Python's stdlib module named `trace`).

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=trace --cov-report=html
```

View coverage report: Open `htmlcov/index.html` in your browser

### Run Specific Test

```bash
pytest tests/test_simulate.py::test_simulate_conflict_data_basic -v
```

## Building Documentation

### Install Documentation Dependencies

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
cd docs
make html
```

### View Documentation

Open `docs/_build/html/index.html` in your web browser.

## Troubleshooting

### Issue: JAX Installation Fails

**Solution**: JAX requires specific versions for different platforms.

**For CPU-only (Linux/macOS):**
```bash
pip install --upgrade "jax[cpu]"
```

**For CUDA (GPU support):**
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: NumPyro Import Error

**Solution**: Ensure JAX is installed first:
```bash
pip install jax jaxlib
pip install numpyro
```

### Issue: Tests Fail with "Module not found"

**Solution**: Install in editable mode:
```bash
pip install -e .
```

### Issue: Matplotlib Display Issues

**Solution**: Set backend before importing:
```python
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt
```

### Issue: MCMC is Very Slow

**Solutions**:
1. **Reduce samples**: Use `num_warmup=500, num_samples=1000` for testing
2. **Use GPU**: Install JAX with CUDA support
3. **Reduce data size**: Test with smaller datasets first

## Verification Checklist

- [ ] Package imports successfully
- [ ] Basic example runs without errors
- [ ] Tests pass
- [ ] Documentation builds
- [ ] Visualizations are generated

## Getting Help

If you encounter issues:

1. **Check the documentation**: `docs/_build/html/index.html`
2. **Review examples**: `examples/README.md`
3. **Search issues**: https://github.com/OJWatson/trace/issues
4. **Open new issue**: Include error messages and system info
5. **Get help**: https://github.com/OJWatson/trace/issues

## System Information

To help with debugging, collect system information:

```python
import sys
import platform
import jax
import numpyro
import numpy as np

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"JAX: {jax.__version__}")
print(f"NumPyro: {numpyro.__version__}")
print(f"NumPy: {np.__version__}")
print(f"JAX devices: {jax.devices()}")
```

## Next Steps

Once installation is verified:

1. **Explore examples**: `examples/01_basic_usage.py`
2. **Read model background**: `docs/model/00_overview.md` and `docs/model/01_background.md`
3. **Run tutorials**: `docs/tutorials/01_basic_example.ipynb` and `docs/tutorials/02_gaza_analysis.ipynb`
4. **Try with your data**: Follow the data preparation guide
5. **Contribute**: See `CONTRIBUTING.md`

## Performance Tips

### For Faster Inference

1. **Use GPU**: Install JAX with CUDA support
2. **Increase chains**: Use `num_chains=4` for parallel sampling
3. **Optimize warmup**: Start with `num_warmup=500` and increase if needed
4. **Use JAX JIT**: Models are already JIT-compiled

### For Large Datasets

1. **Subsample for testing**: Use smaller date ranges initially
2. **Increase memory**: Close other applications
3. **Use batch processing**: Process regions separately if needed

## Uninstallation

To remove TRACE:

```bash
pip uninstall trace-conflict
```

To remove virtual environment:

```bash
deactivate  # Exit virtual environment
rm -rf venv  # Remove virtual environment directory
```

---

**Last Updated**: December 2024  
**Package Version**: 0.1.0  
**Maintainer**: Imperial College London
