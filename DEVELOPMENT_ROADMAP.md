# TRACE Development Roadmap

**Last Updated**: December 16, 2025  
**Current Version**: 0.1.0

---

N.B. Claude Opus Generated Next Steps based on dictated goals. 

## Objectives (Short-Term Forecasting + Decision Support)

The near-term goal for a student is to extend TRACE from a working proof-of-concept into a forecasting and decision-support tool that:

1. Produces short-term probabilistic forecasts of deaths and hospital incidence
2. Demonstrates whether ACLED provides **additional forecast lead time** (e.g., “a few days”) compared to purely autoregressive baselines
3. Supports scenario analysis (ceasefire / escalation / intervention)
4. Moves beyond total hospital incidence to **hospital incidence typology** (case mix)

This document focuses on research questions, engineering milestones, and evaluation criteria to guide iterative development.

### Recommended development workflow

- Implement one model change at a time
- Add a minimal evaluation / backtest to justify the change
- Keep tutorials updated (especially `docs/tutorials/02_gaza_analysis.ipynb`) so the repo remains runnable and teachable

---

## Priority Milestones (8–12 weeks)

### Milestone A — Forecasting evaluation and baselines

- Define forecast targets:
  - deaths (national)
  - total injured (across hospitals)
  - later: hospital-level, and typology
- Create a rolling-origin evaluation:
  - train on days `1..t`, forecast `t+1..t+h`
  - horizons: 1, 3, 7, 14 days
- Baseline comparators:
  - persistence / moving-average
  - Negative Binomial GLM with lagged deaths only
  - Negative Binomial GLM with lagged deaths + lagged events

### Milestone B — Event typology and severity features

- Extend event input beyond raw count:
  - counts by event type (ACLED `event_type`)
  - severity proxies (e.g. ACLED reported fatalities, if usable)
  - spatial proximity to hospitals / population centers
- Evaluate which features improve short-term forecast skill.

### Milestone C — Hierarchical structure

- Implement partial pooling across:
  - regions/governorates (if data available)
  - hospitals
  - event types (effects shared with shrinkage)
- Goal: stabilize inference when data are sparse and improve generalization.

### Milestone D — Hospital incidence typology (case mix)

- Define typology labels (dependent on data availability):
  - injury severity bands
  - mechanism/type proxies (burns vs blast vs gunshot)
  - or simpler operational categories (minor/major/critical)
- Extend the observation model to multi-category counts:
  - multinomial / Dirichlet-multinomial
  - or independent NB by category with shared latent intensity

### Milestone E — Scenario API + reporting

- Standardize scenario definitions:
  - baseline
  - ceasefire
  - escalation
  - custom event trajectory
- Add reporting outputs:
  - forecast plots
  - calibration plots
  - “lead time gained” summary

---

## Research Questions

- Does ACLED improve forecasts compared to autoregressive-only baselines? At what horizons?
- Which ACLED signals matter: incidence, typology, geography, or reported severity?
- Are time-varying casualty rates best modeled by random walks, splines, or latent intensity processes?
- How sensitive are results to reporting artifacts (mortality series discontinuities)?
- Can we infer and forecast typology of hospital admissions, and is it decision-relevant?
- Is there spatial patterns in ACLED data? Can we predict future ACLED events in space-time based on past events?

---

## Current State: What Works ✓

1. **Basic Model**: Single-region casualty model with spatial kernels
2. **Data Integration**: ACLED events, mortality data (Tech for Palestine)
3. **Inference**: MCMC with NUTS, convergence diagnostics
4. **Visualization**: Model fit plots, forecasts
5. **Documentation**: Academic background, tutorials, API reference
6. **Examples/Tutorials**:
   - `docs/tutorials/01_basic_example.ipynb`
   - `docs/tutorials/02_gaza_analysis.ipynb`
7. **Overdispersion**: Implemented via `GammaPoisson` (Negative Binomial)
8. **Time-varying rates (opt-in)**: `casualty_model_random_walk` with log random walks for `mu_w[t]`, `mu_i[t]`

---

## Gaps to add

### 1. ⚠️ Spatial Kernel Visualization MISSING

**Current State**: Spatial kernels are implemented but not visualized

**What's Needed**:
- [ ] Function to plot spatial allocation patterns
- [ ] Visualization of how events allocate to hospitals by distance
- [ ] Interactive plots showing kernel parameters (ell) effect
- [ ] Hospital catchment area visualization
- [ ] Example: "Which hospitals serve which areas?"

**Implementation**:
```python
# New function needed in trace/analysis.py or trace/plotting.py
def plot_spatial_allocation(
    event_coords, 
    hospital_coords, 
    ell, 
    event_weights=None
):
    """
    Visualize how events allocate to hospitals via spatial kernel.
    - Map view with events and hospitals
    - Allocation weights as colors/arrows
    - Kernel decay curves
    """
```

**Tutorial**: Add section in `01_basic_example.ipynb` showing:
- How kernel length scale affects allocation
- Sensitivity to ell parameter
- Multiple scenarios (ell=5, 20, 50)

---

### 2. ⚠️ Multiple Hospital Simulations MINIMAL

**Current State**: `simulate_conflict_data()` creates hospitals but doesn't demonstrate spatial effects well

**What's Needed**:
- [ ] Simulator with explicit hospital capacity constraints
- [ ] Visualization of hospital-specific injury patterns
- [ ] Comparison of simulations with different hospital configurations
- [ ] Example: "What if one hospital closes?"

**Implementation**:
```python
# Enhancement to trace/simulate.py
def simulate_multi_hospital_scenario(
    n_hospitals,
    hospital_capacities,  # NEW
    hospital_locations,   # User-specified, not random
    events,
    ell,
    ...
):
    """
    Simulate casualties with realistic hospital constraints.
    Show how spatial allocation works in practice.
    """
```

**Tutorial**: New section or notebook:
- `03_spatial_analysis.ipynb`
- Show 3-5 hospital configurations
- Plot injuries by hospital over time
- Demonstrate how ell affects distribution
- Show impact of hospital closure/damage

---

### 3. ❌ HIERARCHICAL INFERENCE NOT IMPLEMENTED

**Current State**: Only flat (single-region) models exist

**What's Needed**:
- [ ] Multi-region hierarchical model
- [ ] Partial pooling of casualty rates across regions
- [ ] Region-specific random effects
- [ ] Hyperpriors for hierarchical parameters

**Implementation**:
```python
# New model in trace/model.py
def hierarchical_casualty_model(
    events_by_day_by_region,      # Dict[region_id, array]
    event_coords_by_region,        # Dict[region_id, coords]
    hospital_coords_by_region,     # Dict[region_id, coords]
    injuries_obs_by_region,        # Dict[region_id, matrix]
    deaths_obs_by_region,          # Dict[region_id, array]
    pooling="partial",             # "none", "partial", "full"
    ...
):
    """
    Hierarchical model with partial pooling across regions.
    
    Model structure:
    - Global: mu_w_global, mu_i_global, p_late_global
    - Regional: mu_w_r ~ Normal(mu_w_global, sigma_w)
    - Partial pooling shrinks regional estimates toward global mean
    """
    # Global hyperparameters
    mu_w_global = numpyro.sample("mu_w_global", dist.Exponential(0.3))
    sigma_w = numpyro.sample("sigma_w", dist.Exponential(1.0))
    
    # Regional parameters (partial pooling)
    with numpyro.plate("regions", n_regions):
        mu_w_region = numpyro.sample("mu_w_region", 
                                      dist.Normal(mu_w_global, sigma_w))
    
    # Likelihood for each region
    for r in regions:
        # ... use mu_w_region[r] for region r's casualties
```

**Documentation**:
- New vignette: `model/05_partial_pooling.md`
- Mathematical derivation
- When to use hierarchical vs flat models
- Interpretation of shrinkage

**Tutorial**:
- `04_multi_region_analysis.ipynb`
- Example: Model multiple governorates in Gaza
- Compare pooled vs unpooled estimates
- Show shrinkage towards global mean
- Forest plots of region-specific parameters

---

### 4. ❌ FLEXIBLE MODEL SPECIFICATION NOT IMPLEMENTED

**Current State**: Hard-coded models (`casualty_model`, `casualty_model_with_covariates`)

**Design Decision**: Use **Python builder pattern** instead of R-style formulas
- ✅ JAX-native (JIT-compilable)
- ✅ Type-safe (IDE autocomplete, mypy)
- ✅ Pythonic (chainable methods)
- ✅ Flexible (functional escape hatch for custom models)

**Desired User Experience**:

**Simple Case - Builder Pattern**:
```python
from trace import CasualtyModel

# Convenient builder for common models
model = (
    CasualtyModel()
    .injuries(covariates=['events', 'ceasefire', 'displacement'])
    .deaths(immediate='events', delayed='injuries')
    .spatial('exponential', length_scale=20.0)
    .hierarchical(
        regions=['Gaza City', 'Khan Younis', 'Rafah'],
        partial_pool=['events', 'ceasefire']
    )
    .priors({
        'events': Prior.Normal(5, 2),
        'ceasefire': Prior.Normal(-0.5, 0.2),
        'p_late': Prior.Beta(2, 10)
    })
)

# Fit the model
results = model.fit(data, num_warmup=1000, num_samples=2000)

# Extract results
results.summary()
results.plot_effects()
```

**Advanced Case - Functional Definition**:
```python
# Custom injury rate function for complex interactions
def custom_injury_rate(events, ceasefire, population_density, beta):
    """Nonlinear model with urban effects."""
    conflict_intensity = events * jnp.exp(beta['ceasefire'] * ceasefire)
    urban_modifier = 1 + beta['urban'] * population_density
    return conflict_intensity * urban_modifier

# Builder with functional escape hatch
model = (
    CasualtyModel()
    .injuries(rate_fn=custom_injury_rate)  # Custom function
    .deaths(delayed='injuries')
    .spatial('gaussian', length_scale=15)
)
```

**Implementation** (Staged Approach):

**Stage 1: Core Infrastructure** (1 week)
```python
# New module: trace/priors.py
class Prior:
    """Type-safe prior specification."""
    @staticmethod
    def Normal(mu: float, sigma: float):
        return _PriorSpec("Normal", mu=mu, sigma=sigma)
    
    @staticmethod
    def Exponential(rate: float):
        return _PriorSpec("Exponential", rate=rate)
    
    @staticmethod
    def Beta(alpha: float, beta: float):
        return _PriorSpec("Beta", alpha=alpha, beta=beta)
    
    def to_numpyro(self):
        """Convert to NumPyro distribution."""
        # Map to dist.Normal(), dist.Exponential(), etc.

# New module: trace/specs.py
@dataclass
class InjurySpec:
    covariates: Optional[List[str]]
    rate_fn: Optional[Callable]
    priors: Dict[str, Prior]
    link: str = "log"

@dataclass
class SpatialSpec:
    kernel: str  # "exponential", "gaussian", "powerlaw"
    params: Dict[str, float]

@dataclass
class HierarchicalSpec:
    regions: List[str]
    partial_pool: List[str]
    shared_params: List[str] = field(default_factory=lambda: ['ell'])
```

**Stage 2: Model Builder** (2 weeks)
```python
# New module: trace/builder.py
class CasualtyModel:
    """Builder for casualty models."""
    
    def __init__(self):
        self._injury_spec = None
        self._death_spec = None
        self._spatial_spec = None
        self._hierarchical_spec = None
        self._priors = {}
    
    def injuries(self, 
                 covariates: Optional[List[str]] = None,
                 rate_fn: Optional[Callable] = None):
        """
        Specify injury rate model.
        
        Parameters
        ----------
        covariates : List[str]
            Names of covariates for linear predictor
        rate_fn : Callable
            Custom function(data, params) -> rate for nonlinear models
        """
        if rate_fn is None and covariates is not None:
            # Build default linear predictor
            rate_fn = self._build_linear_predictor(covariates)
        
        self._injury_spec = InjurySpec(covariates, rate_fn, {})
        return self  # Chainable
    
    def deaths(self, immediate=None, delayed=None):
        """Specify death model."""
        self._death_spec = DeathSpec(immediate, delayed)
        return self
    
    def spatial(self, kernel: str, **params):
        """Spatial kernel configuration."""
        self._spatial_spec = SpatialSpec(kernel, params)
        return self
    
    def hierarchical(self, regions: List[str], partial_pool: List[str]):
        """Hierarchical structure."""
        self._hierarchical_spec = HierarchicalSpec(regions, partial_pool)
        return self
    
    def priors(self, prior_dict: Dict[str, Prior]):
        """Set priors for parameters."""
        self._priors.update(prior_dict)
        return self
    
    def build(self) -> Callable:
        """
        Build NumPyro model function from specifications.
        
        Returns
        -------
        Callable
            NumPyro model function that can be passed to MCMC
        """
        self._validate()  # Check all required specs are set
        
        if self._hierarchical_spec is None:
            return self._build_flat_model()
        else:
            return self._build_hierarchical_model()
    
    def fit(self, data, num_warmup=1000, num_samples=2000, **kwargs):
        """Fit model to data."""
        model_fn = self.build()
        kernel = NUTS(model_fn)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(PRNGKey(0), data=data)
        return ModelResults(mcmc, self)
```

**Stage 3: Result Objects** (1 week)
```python
# New module: trace/results.py
class ModelResults:
    """Wrapper around MCMC results with convenient methods."""
    
    def __init__(self, mcmc, model_spec):
        self.mcmc = mcmc
        self.samples = mcmc.get_samples()
        self.model_spec = model_spec
    
    def summary(self, params: Optional[List[str]] = None):
        """Print parameter summary table."""
        # Pretty-print with parameter names from model_spec
    
    def plot_effects(self, covariate: str):
        """Plot marginal effect of a covariate."""
        # Show how outcome changes with covariate
    
    def plot_trace(self):
        """Trace plots for convergence."""
        # Wrapper around ArviZ
    
    def posterior_predictive(self):
        """Generate posterior predictive samples."""
        # Use model_spec to generate predictions
    
    def to_arviz(self):
        """Convert to ArviZ InferenceData."""
        return az.from_numpyro(self.mcmc)
```

**Documentation**:
- New vignette: `model/06_flexible_specification.md`
  - Builder pattern overview
  - Common model specifications
  - Functional escape hatch for custom models
  - Prior specification
  - Comparison to hard-coded models
- Tutorial: `05_advanced_modeling.ipynb`
  - Builder pattern examples
  - Adding covariates (ceasefire, media coverage, aid access)
  - Custom injury rate functions
  - Hierarchical models
  - Prior sensitivity analysis

**Timeline**: 4-5 weeks total (1 week infra, 2 weeks builder, 1 week results, 1 week docs)

---

### 5. ⚠️ PLOTTING CAPABILITIES LIMITED

**Current State**: Basic fit plots, forecast plots

**What's Needed**:

#### Spatial Plots
- [ ] Event density heatmaps
- [ ] Hospital catchment areas
- [ ] Spatial allocation visualizations
- [ ] Conflict intensity over geographic regions

#### Temporal Plots
- [ ] Time series with events overlay
- [ ] Injury-to-death delay visualization
- [ ] Rolling averages and trends
- [ ] Intervention effects (before/after)

#### Diagnostic Plots
- [ ] Prior vs posterior distributions
- [ ] Pair plots for parameter correlations
- [ ] Trace plots (already in ArviZ but need wrapper)
- [ ] LOO-CV / WAIC model comparison

#### Effect Plots
- [ ] Coefficient plots (forest plots)
- [ ] Marginal effects of covariates
- [ ] Interaction plots
- [ ] Hierarchical shrinkage plots

**Implementation**:
```python
# New module: trace/plotting.py (expand current)

def plot_spatial_distribution(events, hospitals, intensity_grid):
    """Heatmap of conflict intensity with hospital locations."""
    
def plot_temporal_patterns(dates, events, injuries, deaths, interventions=None):
    """Multi-panel time series with intervention markers."""
    
def plot_delay_distribution(injuries, deaths, estimated_delays):
    """Visualize injury-to-death delay dynamics."""
    
def plot_parameter_forest(samples, param_names, regions=None):
    """Forest plot for hierarchical parameters."""
    
def plot_marginal_effects(model, covariate, data):
    """Show how outcome changes with covariate (holding others constant)."""
```

---

### 6. ⚠️ PRIOR SPECIFICATION INFLEXIBLE

**Current State**: Priors hard-coded in model functions

**What's Needed**:
- [ ] User-specifiable priors via API
- [ ] Prior predictive checks
- [ ] Sensitivity analysis utilities

**Implementation**:
```python
# New module: trace/priors.py
class Prior:
    """Represent a prior distribution."""
    def __init__(self, family, **params):
        self.family = family  # "Normal", "Exponential", "Beta", etc.
        self.params = params
    
    def to_numpyro(self):
        """Convert to NumPyro distribution object."""
        if self.family == "Normal":
            return dist.Normal(self.params['mu'], self.params['sigma'])
        elif self.family == "Exponential":
            return dist.Exponential(self.params['rate'])
        # ... etc

# Usage in model builder
priors = {
    'mu_w': Prior("Normal", mu=5, sigma=2),      # Informative
    'mu_i': Prior("Exponential", rate=0.5),      # Weakly informative
    'p_late': Prior("Beta", alpha=2, beta=10)
}
```

**Utilities**:
```python
def prior_predictive_check(model, priors, n_samples=1000):
    """
    Sample from prior, generate predictions, check reasonableness.
    Returns plots comparing prior predictive to typical data ranges.
    """

def sensitivity_analysis(model, data, prior_variants):
    """
    Fit model with different priors, compare posterior estimates.
    Show how sensitive inference is to prior choices.
    """
```

---

## Implementation Priority

### Phase 1: Essential Features (1-2 weeks)
1. **Spatial visualization** - Understand what the model is doing
2. **Multi-hospital simulations** - Better validation
3. **Plotting enhancements** - Spatial/temporal data viz

### Phase 2: Hierarchical Models (2-3 weeks)
4. **Partial pooling implementation** - Multi-region models
5. **Hierarchical documentation** - Vignette + tutorial
6. **Example with Gaza data** - Multiple governorates

### Phase 3: Flexible API (3-4 weeks)
7. **Formula parsing** - R-style syntax
8. **Model builder** - Dynamic model construction
9. **High-level API** - TraceModel class
10. **Prior specification** - Flexible priors

### Phase 4: Advanced Features (2-3 weeks)
11. **Prior predictive checks** - Validation tools
12. **Sensitivity analysis** - Robustness testing
13. **Model comparison** - LOO-CV, WAIC
14. **Effect plots** - Marginal effects, interactions

---

## Detailed Task Breakdown

### Task 1: Spatial Visualization (High Priority)

**Files to Create/Modify**:
- `src/trace/plotting.py` - Add spatial plot functions
- `docs/tutorials/01_basic_example.ipynb` - Add spatial section
- `docs/tutorials/03_spatial_analysis.ipynb` - New tutorial (optional)

**Functions Needed**:
```python
def plot_spatial_kernel(ell, max_distance=100):
    """Plot kernel decay curve."""
    
def plot_event_hospital_allocation(events, hospitals, ell, weights):
    """Map showing allocation arrows from events to hospitals."""
    
def plot_hospital_catchment_areas(hospitals, ell, grid_resolution=50):
    """Heatmap of which areas are served by which hospital."""
    
def compare_kernel_parameters(events, hospitals, ell_values):
    """Side-by-side comparison of different ell values."""
```

**Estimated Time**: 3-4 days

---

### Task 2: Hierarchical Model (High Priority)

**Files to Create/Modify**:
- `src/trace/model.py` - Add `hierarchical_casualty_model()`
- `src/trace/analysis.py` - Add `run_hierarchical_inference()`
- `docs/model/05_partial_pooling.md` - New vignette
- `docs/tutorials/04_multi_region.ipynb` - New tutorial

**Model Structure**:
```python
def hierarchical_casualty_model(
    regions: List[str],
    events_by_region: Dict[str, np.ndarray],
    hospitals_by_region: Dict[str, np.ndarray],
    injuries_by_region: Dict[str, np.ndarray],
    deaths_by_region: Dict[str, np.ndarray],
    pooling: str = "partial",  # "none", "partial", "full"
    shared_params: List[str] = ["ell"],  # Which params to share vs pool
    ...
):
    n_regions = len(regions)
    
    # Global hyperparameters
    mu_w_global = numpyro.sample("mu_w_global", dist.Exponential(0.3))
    sigma_w = numpyro.sample("sigma_w", dist.Exponential(1.0))
    
    mu_i_global = numpyro.sample("mu_i_global", dist.Exponential(0.3))
    sigma_i = numpyro.sample("sigma_i", dist.Exponential(1.0))
    
    # Shared spatial parameter (no pooling needed)
    ell = numpyro.sample("ell", dist.Exponential(1.0))
    
    # Regional parameters with partial pooling
    with numpyro.plate("regions", n_regions):
        mu_w_region = numpyro.sample("mu_w_region", 
                                      dist.Normal(mu_w_global, sigma_w))
        mu_i_region = numpyro.sample("mu_i_region",
                                      dist.Normal(mu_i_global, sigma_i))
        p_late_region = numpyro.sample("p_late_region",
                                        dist.Beta(2, 10))
    
    # Likelihood for each region
    for r, region_name in enumerate(regions):
        events = events_by_region[region_name]
        hospitals = hospitals_by_region[region_name]
        
        # Use region-specific parameters
        # ... (similar to flat model but with mu_w_region[r], etc.)
```

**Documentation Structure** (`05_partial_pooling.md`):
1. Introduction - Why hierarchical models?
2. No pooling (separate models per region)
3. Complete pooling (single shared model)
4. Partial pooling (best of both worlds)
5. Mathematical formulation
6. Shrinkage and borrowing strength
7. When to use which approach
8. Implementation in TRACE
9. Interpretation of hyperparameters

**Estimated Time**: 5-7 days

---

### Task 3: Formula-Based API (Medium Priority, Complex)

**Phase 3a: Foundation (1 week)**
- Design API
- Formula parsing with patsy
- Basic model builder

**Phase 3b: Implementation (2 weeks)**
- Dynamic model construction
- Prior specification
- Hierarchical structure from formula
- Testing

**Phase 3c: Documentation (1 week)**
- Vignette on formula syntax
- Tutorial with examples
- API reference

**Estimated Time**: 3-4 weeks total

---

## Success Criteria

### Spatial Visualization ✓
- [ ] Users can visualize how events allocate to hospitals
- [ ] Kernel parameter effects are clear
- [ ] Sensitivity to ell is demonstrable

### Hierarchical Models ✓
- [ ] Multi-region model implemented
- [ ] Partial pooling works correctly
- [ ] Shrinkage visualized
- [ ] Documentation complete

### Formula API ✓
- [ ] Users can specify models via formulas
- [ ] Priors customizable
- [ ] Hierarchical structure specifiable
- [ ] Documentation with examples

### Overall Package Quality ✓
- [ ] All tests pass
- [ ] Documentation comprehensive
- [ ] Tutorials cover new features
- [ ] Examples demonstrate capabilities

---

## References for Implementation

### Hierarchical Models
- **Gelman & Hill (2006)**: Data Analysis Using Regression and Multilevel/Hierarchical Models
- **McElreath (2020)**: Statistical Rethinking (Ch. 13-14 on multilevel models)
- **epidemia vignettes**: Partial pooling examples

### Formula Interfaces
- **patsy documentation**: https://patsy.readthedocs.io/
- **bambi source code**: Bayesian formula interface on top of PyMC
- **epidemia source code**: R formula interface

### Spatial Methods
- **Gething et al. (2011)**: Spatial modeling with kernel methods
- **Banerjee et al. (2014)**: Hierarchical Modeling and Analysis for Spatial Data

---

## Notes

- Formula API is most complex feature - defer to Phase 3
- Spatial visualization needed ASAP for understanding/debugging
- Hierarchical models are next priority for real applications
- All features should have:
  - Unit tests
  - Documentation (vignette)
  - Tutorial (notebook with real example)
  - API reference

**Last Updated**: December 4, 2024
