# Model Implementation

This vignette describes the NumPyro implementation of the **TRACE** casualty model. We explain the code structure, demonstrate how to specify models, and provide guidance on computational considerations.

## 1. NumPyro Primer

[NumPyro](https://num.pyro.ai) is a lightweight probabilistic programming library built on JAX. It provides:

- **Expressive modeling language**: Define complex hierarchical models declaratively
- **Efficient inference**: State-of-the-art MCMC algorithms (NUTS, HMC)
- **GPU acceleration**: Leverage JAX's JIT compilation and device support
- **Automatic differentiation**: Enable gradient-based inference algorithms

### 1.1 Basic Model Structure

A NumPyro model is a Python function that defines the generative process:

```python
import numpyro
import numpyro.distributions as dist

def simple_model(obs=None):
    # Sample from prior
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    
    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)
```

Key concepts:

- `numpyro.sample()`: Declares a random variable
- First argument: Parameter name (used in MCMC output)
- Second argument: Distribution
- `obs` keyword: Conditions on observed data (if provided)

### 1.2 Inference with NUTS

To fit the model:

```python
from numpyro.infer import MCMC, NUTS
import jax

# Define sampler
kernel = NUTS(simple_model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)

# Run inference
mcmc.run(jax.random.PRNGKey(0), obs=data)

# Extract samples
samples = mcmc.get_samples()
```

Parameters:

- `num_warmup`: Burn-in iterations for adaptation
- `num_samples`: Posterior samples to draw
- `num_chains`: Number of independent chains (for convergence assessment)

## 2. TRACE Model Implementation

The core model is implemented in `trace.model.casualty_model()`. We'll walk through each component.

### 2.1 Function Signature

```python
def casualty_model(
    events_by_day: np.ndarray,      # (T,) event counts per day
    event_day_index: np.ndarray,    # (E,) day index for each event
    event_coords: np.ndarray,        # (E, 2) event locations
    hospital_coords: np.ndarray,     # (H, 2) hospital locations
    injuries_obs: np.ndarray,        # (T, H) observed injuries
    deaths_obs: np.ndarray,          # (T,) observed deaths
    delay_probs: Optional[np.ndarray] = None,  # (L,) delay distribution
) -> None:
```

**Inputs**:

- `events_by_day`: Number of events on each day (length $T$)
- `event_day_index`: Day index (0-indexed) for each of the $E$ events
- `event_coords`: $(x, y)$ coordinates for each event
- `hospital_coords`: $(x, y)$ coordinates for each of $H$ hospitals
- `injuries_obs`: Observed injury counts (matrix of size $T \times H$)
- `deaths_obs`: Observed daily deaths (vector of length $T$)
- `delay_probs`: Delay distribution $f_k$ (defaults to [0.5, 0.3, 0.15, 0.05])

**Returns**: None (model functions define the probabilistic program, don't return values)

### 2.2 Parameter Priors

```python
# Average wounded per event
mu_w = numpyro.sample("mu_w", dist.Exponential(0.3))

# Average immediate deaths per event
mu_i = numpyro.sample("mu_i", dist.Exponential(0.3))

# Hospital fatality rate
p_late = numpyro.sample("p_late", dist.Beta(2, 10))

# Spatial length scale
ell = numpyro.sample("ell", dist.Exponential(1.0))
```

Each `numpyro.sample()` call:

1. Names the parameter (e.g., "mu_w")
2. Specifies the prior distribution
3. Returns a sample (during inference) or prior draw (during prior predictive sampling)

### 2.3 Spatial Kernel

```python
def spatial_kernel_weights(
    event_coords: jnp.ndarray,
    hospital_coords: jnp.ndarray,
    ell: float
) -> jnp.ndarray:
    """Compute normalized spatial weights."""
    # Pairwise distances: (E, H)
    dists = jnp.linalg.norm(
        event_coords[:, None, :] - hospital_coords[None, :, :],
        axis=2
    )
    
    # Exponential decay kernel
    weights = jnp.exp(-dists / ell)
    
    # Normalize to sum to 1 for each event
    norm_weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-8)
    
    return norm_weights  # (E, H)
```

**Broadcasting**:

- `event_coords[:, None, :]`: Shape (E, 1, 2)
- `hospital_coords[None, :, :]`: Shape (1, H, 2)
- Subtraction broadcasts to (E, H, 2)
- `linalg.norm(..., axis=2)` reduces to (E, H) distance matrix

**Numerical Stability**: Adding `1e-8` prevents division by zero if all weights are negligible.

### 2.4 Hospital Injury Likelihood

```python
# Compute spatial weights for all events
norm_weights = spatial_kernel_weights(
    jnp.array(event_coords),
    jnp.array(hospital_coords),
    ell
)

# Accumulate event contributions per day
effective_events = jnp.zeros((T, H))
effective_events = effective_events.at[event_day_index].add(norm_weights)

# Expected injuries
lam_injuries = mu_w * effective_events  # (T, H)
lam_injuries = jnp.maximum(lam_injuries, 1e-8)  # Numerical stability

# Likelihood
numpyro.sample(
    "obs_injuries",
    dist.Poisson(lam_injuries).to_event(2),
    obs=jnp.array(injuries_obs)
)
```

**Key Operations**:

1. `spatial_kernel_weights()`: Returns (E, H) matrix of allocation weights
2. `.at[event_day_index].add()`: JAX's functional array update — accumulates weights by day
3. `to_event(2)`: Treats the (T, H) matrix as a single multivariate observation
4. `obs=...`: Conditions on observed data

**Why `to_event(2)`?** Without it, each $H_{t,j}$ would be treated as an independent observation. `to_event(2)` declares that the entire matrix is a single realization of the model.

### 2.5 Death Likelihood

```python
# Total injuries per day
injuries_total = jnp.nansum(jnp.array(injuries_obs), axis=1)  # (T,)

# Convolve with delay distribution
L = delay_probs.shape[0]
pad = jnp.pad(injuries_total, (0, L))
conv_deaths = jnp.zeros(T)

for k in range(L):
    # Deaths k+1 days after injury
    conv_deaths = conv_deaths + pad[k : T + k] * delay_probs[k]

# Expected deaths
expected_immediate = mu_i * jnp.array(events_by_day)
expected_late = p_late * conv_deaths
expected_deaths = expected_immediate + expected_late
expected_deaths = jnp.maximum(expected_deaths, 1e-8)

# Likelihood
numpyro.sample(
    "obs_deaths",
    dist.Poisson(expected_deaths),
    obs=jnp.array(deaths_obs)
)
```

**Convolution Details**:

- `pad(injuries_total, (0, L))`: Pads array on the right to avoid index errors
- Loop over delay lags: Each iteration adds contribution from injuries $k$ days ago
- Result: $\sum_{k=1}^L I_{t-k} f_k$ for each day $t$

**Why not use `jnp.convolve()`?** The standard convolution function doesn't handle the temporal indexing we need (deaths from past injuries), so we implement it manually with a loop.

## 3. Running Inference

### 3.1 Preparing Data

```python
from trace.data import load_example_acled_data, prepare_acled_events

# Load ACLED data
events_df = load_example_acled_data()

# Prepare for modeling
events_by_day, event_days, event_coords, dates = prepare_acled_events(
    events_df,
    start_date="2023-10-07",
    end_date="2023-12-31"
)

# Hospital coordinates (example)
hospital_coords = np.array([
    [31.5, 34.45],   # Gaza City
    [31.35, 34.30],  # Khan Younis
    [31.28, 34.25],  # Rafah
])

# Load injuries and deaths (from your data sources)
injuries_obs = load_hospital_data("hospital_data.csv")
deaths_obs = load_mortality_data("deaths.csv")
```

### 3.2 Specifying Priors

Default priors are built into the model, but you can modify them by editing `trace.model.casualty_model()`:

```python
# More informative prior based on domain knowledge
mu_w = numpyro.sample("mu_w", dist.Normal(5.0, 1.0))  # Mean 5, SD 1

# Tighter prior on hospital fatality rate
p_late = numpyro.sample("p_late", dist.Beta(5, 20))  # Mean 0.2, tighter
```

### 3.3 Running MCMC

```python
from trace.analysis import run_inference

mcmc, samples = run_inference(
    events_by_day=events_by_day,
    event_day_index=event_days,
    event_coords=event_coords,
    hospital_coords=hospital_coords,
    injuries_obs=injuries_obs,
    deaths_obs=deaths_obs,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    rng_seed=42
)
```

**Recommended Settings**:

- **Development**: `num_warmup=500, num_samples=1000, num_chains=1` (fast iteration)
- **Production**: `num_warmup=2000, num_samples=4000, num_chains=4` (robust inference)
- **Publication**: `num_warmup=5000, num_samples=10000, num_chains=4` (high precision)

### 3.4 Accessing Results

```python
# Posterior means
print(f"mu_w: {samples['mu_w'].mean():.2f}")
print(f"mu_i: {samples['mu_i'].mean():.2f}")
print(f"p_late: {samples['p_late'].mean():.3f}")
print(f"ell: {samples['ell'].mean():.2f}")

# Credible intervals
mu_w_ci = np.percentile(samples['mu_w'], [2.5, 97.5])
print(f"95% CI for mu_w: [{mu_w_ci[0]:.2f}, {mu_w_ci[1]:.2f}]")

# Full summary
mcmc.print_summary()
```

## 4. Computational Considerations

### 4.1 Performance Optimization

**GPU Acceleration**: If a CUDA-compatible GPU is available:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

JAX will automatically use the GPU, providing 10-100x speedup.

**Multiple Chains in Parallel**:

```python
import numpyro
numpyro.set_host_device_count(4)  # Use 4 CPU cores for parallel chains
```

**JIT Compilation**: NumPyro automatically JIT-compiles the model. First run will be slow (compilation), subsequent runs are fast.

### 4.2 Memory Management

For large datasets:

- **Subsample**: Fit model to a subset of time period initially
- **Batch Processing**: Process multiple regions separately, then combine
- **Thinning**: Keep every $k$-th MCMC sample to reduce memory

### 4.3 Convergence Diagnostics

Check convergence after every run:

```python
import arviz as az

# Convert to ArviZ format
idata = az.from_numpyro(mcmc)

# Diagnostics
print(az.summary(idata))

# Rhat should be < 1.01
assert (az.rhat(idata) < 1.01).all(), "Chains have not converged!"

# Effective sample size should be > 100 per chain
assert (az.ess(idata) > 100).all(), "Insufficient effective samples!"
```

**Trace Plots**:

```python
az.plot_trace(idata)
plt.tight_layout()
plt.savefig("trace_plot.png")
```

Look for:

- **Stationarity**: No trends or drift
- **Mixing**: Rapid exploration of parameter space
- **Between-chain agreement**: Multiple chains should overlap

## 5. Posterior Predictive Sampling

Generate predictions from the posterior:

```python
from trace.analysis import posterior_predictive

preds = posterior_predictive(
    samples=samples,
    events_by_day=events_by_day,
    event_day_index=event_days,
    event_coords=event_coords,
    hospital_coords=hospital_coords,
    injuries_obs_shape=injuries_obs.shape,
    deaths_obs_shape=len(deaths_obs),
    delay_probs=delay_probs,
    rng_seed=1
)

# preds['obs_injuries']: (num_samples, T, H)
# preds['obs_deaths']: (num_samples, T)

# Compute credible intervals
deaths_median = np.median(preds['obs_deaths'], axis=0)
deaths_lower = np.percentile(preds['obs_deaths'], 2.5, axis=0)
deaths_upper = np.percentile(preds['obs_deaths'], 97.5, axis=0)
```

## 6. Forecasting

Project future casualties under scenarios:

```python
from trace.analysis import forecast

# Scenario: Continue with 2 events per day
future_events = np.full(30, 2.0)

forecast_results = forecast(
    samples=samples,
    future_events_by_day=future_events,
    delay_probs=delay_probs
)

# Results contain median and credible intervals
print(forecast_results['deaths_median'])
print(forecast_results['deaths_lower'])
print(forecast_results['deaths_upper'])
```

## 7. Model Extensions

### 7.1 Time-Varying Parameters

Modify the model to include covariates:

```python
from trace.model import casualty_model_with_covariates

# Define intervention indicator
ceasefire = np.zeros(T)
ceasefire[60:90] = 1  # Ceasefire days 60-90

covariates = ceasefire.reshape(-1, 1)

mcmc, samples = run_inference_with_covariates(
    ...,
    covariates=covariates
)

# Coefficient interpretation
beta = samples['beta'][:, 0]  # Effect of ceasefire
print(f"Ceasefire effect: {np.exp(beta.mean()):.2f}x multiplier")
```

### 7.2 Multiple Regions

For hierarchical models across regions, see [Partial Pooling](05_partial_pooling.md).

### 7.3 Negative Binomial

To handle overdispersion, replace Poisson with Negative Binomial in the model code:

```python
# Instead of:
# numpyro.sample("obs_deaths", dist.Poisson(expected_deaths), obs=deaths_obs)

# Use:
phi = numpyro.sample("phi_deaths", dist.Gamma(0.5, 0.5))
numpyro.sample(
    "obs_deaths",
    dist.NegativeBinomial2(mean=expected_deaths, concentration=phi),
    obs=deaths_obs
)
```

## 8. Debugging Tips

### 8.1 Prior Predictive Checks

Sample from the prior (before seeing data):

```python
from numpyro.infer import Predictive

prior_pred = Predictive(casualty_model, num_samples=1000)
prior_samples = prior_pred(
    jax.random.PRNGKey(0),
    events_by_day=events_by_day,
    event_day_index=event_days,
    event_coords=event_coords,
    hospital_coords=hospital_coords,
    injuries_obs=None,  # Don't condition on data
    deaths_obs=None,
    delay_probs=delay_probs
)

# Check if prior predictions are reasonable
print(f"Prior predictive deaths: {prior_samples['obs_deaths'].mean():.1f}")
```

If prior predictions are wildly unrealistic, adjust priors.

### 8.2 Initialization Failures

If you see "initial energy is NaN" or similar errors:

1. **Check data**: Ensure no NaN or inf values
2. **Numerical stability**: Add small epsilon to rates (already done in code)
3. **Prior support**: Ensure priors match parameter constraints
4. **Initialization**: Try different random seeds

### 8.3 Divergences

If MCMC reports divergent transitions:

1. **Increase target acceptance**: `target_accept_prob=0.9` (default is 0.8)
2. **Reparameterize**: Use non-centered parameterizations for hierarchical models
3. **Stronger priors**: More informative priors can help with difficult geometries

Example:

```python
kernel = NUTS(casualty_model, target_accept_prob=0.95)
```

## 9. Code Reference

All model code is available in:

- **Core model**: `src/trace/model.py`
- **Inference wrapper**: `src/trace/analysis.py`
- **Data preparation**: `src/trace/data.py`
- **Simulation**: `src/trace/simulate.py`

See the [API Documentation](../api/model.md) for complete function signatures.

## References

- **NumPyro Documentation**: https://num.pyro.ai
- **JAX Documentation**: https://jax.readthedocs.io
- **NUTS Algorithm** {cite}`Hoffman2014`
- **Bayesian Workflow** {cite}`Gelman2020`

---

*Next: [Model Schematic](04_schematic.md) provides visual diagrams of the model structure.*
