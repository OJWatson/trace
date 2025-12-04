# Model Introduction

---

The open-source Python package **TRACE** (Temporal and Regional Analysis of Conflict Events) provides a framework for Bayesian, regression-oriented modeling of conflict casualty dynamics. These models are fit to multiple data streams including conflict event data (e.g., from [ACLED](https://acleddata.com)), hospital admission records, and national mortality time series. Casualty dynamics are described through a semi-mechanistic process linking observable data to latent conflict intensity, spatial allocation of casualties to hospitals, and temporal delays from injury to death.

The modeling framework has been motivated by semi-mechanistic epidemic models developed for COVID-19 {cite}`Flaxman2020, Bhatt2020, Mezi_2020`. While traditional conflict modeling has relied on either purely statistical approaches or deterministic agent-based models, **TRACE** occupies a middle ground—explicitly modeling the generative process of casualties while remaining statistically grounded and computationally tractable for Bayesian inference.

## Background and Motivation

Traditional approaches to conflict casualty modeling can be broadly classified as either mechanistic or statistical {cite}`Myers2000`. Mechanistic approaches, such as Lanchester equations {cite}`Lanchester1914` or agent-based models {cite}`Kress2025`, derive casualty dynamics from theoretical considerations about how conflicts unfold. These models are highly interpretable but often difficult to fit to real data and may rely on unrealistic assumptions about combat dynamics.

Purely statistical models, by contrast, make few assumptions about the underlying generative mechanism. Examples include time series models (ARIMA, exponential smoothing), generalized linear models, or modern machine learning approaches. While flexible and often providing good short-term forecasts, these models lack mechanistic interpretation and may give unrealistic projections when extrapolating beyond observed regimes.

**TRACE** implements *semi-mechanistic* models that explicitly describe conflict casualty processes while remaining amenable to statistical inference. The approach has three key components:

1. **Latent Conflict Intensity**: Modeled as a time series that governs the rate at which casualties occur, analogous to the reproduction number $R_t$ in epidemic models
2. **Spatial Structure**: Casualties are allocated to hospitals based on distance-dependent kernels, reflecting accessibility and proximity to conflict events
3. **Temporal Delays**: Deaths occur with delays after initial injury, modeled through discrete delay distributions informed by medical literature

This framework allows for several important capabilities:

- **Multi-stream Data Integration**: Simultaneously modeling conflict events, hospital admissions, and mortality data provides stronger inference than analyzing each in isolation
- **Spatial Modeling**: Understanding how casualties are distributed geographically improves resource allocation and humanitarian planning
- **Scenario-Based Forecasting**: Projecting future casualties under different assumptions (e.g., ceasefire, escalation) informs policy decisions
- **Uncertainty Quantification**: Bayesian inference provides full posterior distributions, allowing principled communication of uncertainty

## Conflict Casualty Dynamics as a Semi-Mechanistic Process

Traditional epidemic models track transitions between epidemiological states (Susceptible → Exposed → Infected → Recovered). In conflict settings, we can draw a useful analogy:

- **Population at Risk** (analogous to Susceptible): Civilians in conflict-affected areas
- **Conflict Events**: Discrete incidents that cause casualties (analogous to exposures)
- **Injured** (analogous to Infected): Individuals admitted to hospitals for conflict-related injuries
- **Deaths** (analogous to severe outcomes): Fatalities, either immediate or after hospitalization

However, conflict dynamics differ in important ways:

1. **Exogenous Forcing**: Unlike infectious diseases where infections self-propagate, conflict intensity is often driven by external factors (military operations, ceasefires, interventions)
2. **Discrete Events**: Casualties arise from discrete conflict events rather than continuous transmission
3. **Spatial Heterogeneity**: Distance to hospitals and conflict locations plays a crucial role
4. **Multiple Pathways**: Deaths can occur immediately at the scene or after varying delays in hospital

The **TRACE** model captures these features while maintaining computational tractability for Bayesian inference using modern probabilistic programming tools ([NumPyro](https://num.pyro.ai)).

## The Bayesian Approach

Bayesian methods offer several advantages for conflict casualty modeling:

### Identifiability and Regularization

Several aspects of casualty models are fundamentally unidentified {cite}`Roosa2019`. For example:

- **Under-reporting**: Observed casualties may represent only a fraction of true casualties, with unknown ascertainment rates
- **Attribution**: Delayed deaths could be attributed to recent injuries or earlier untreated wounds
- **Spatial Ambiguity**: Without precise event locations, casualties could reasonably be allocated to multiple hospitals

Bayesian approaches using MCMC allow full exploration of posterior correlations between coupled parameters. Informative or weakly informative priors can be incorporated to regularize estimation and help mitigate identifiability problems {cite}`Gelman2008, Gelman2013`.

### Uncertainty Quantification

Unlike point estimates, Bayesian inference provides full posterior distributions for all parameters. This is critical for:

- **Forecasting**: Communicating forecast uncertainty through credible intervals
- **Decision-Making**: Understanding the range of plausible outcomes under different scenarios
- **Model Comparison**: Rigorously comparing alternative model specifications

### Hierarchical Modeling

Bayesian hierarchical models allow partial pooling of information across:

- **Multiple Regions**: Sharing information about casualty rates while allowing regional variation
- **Time Periods**: Borrowing strength across similar time periods
- **Event Types**: Partially pooling effects of different conflict types

This is particularly valuable when data for individual regions or periods are sparse.

## Uses of TRACE

**TRACE** can be used for several purposes in conflict analysis:

### 1. Parameter Estimation

Fit the model to observed data to estimate:

- Average casualties per conflict event (wounded and immediate deaths)
- Hospital fatality rates
- Spatial allocation patterns
- Effects of interventions or escalations

### 2. Model Assessment

Evaluate whether a parsimonious semi-mechanistic model can replicate observed patterns. Posterior predictive checks assess model fit, and systematic deviations suggest areas for model extension (e.g., time-varying parameters, overdispersion, covariates).

### 3. Forecasting

Generate probabilistic forecasts of future casualties under different scenarios:

- **Baseline**: Continue current conflict intensity
- **Ceasefire**: Dramatic reduction in conflict events
- **Escalation**: Increased event frequency
- **Intervention**: Specific policy changes (e.g., humanitarian corridors)

Short-term forecasts inform resource allocation (hospital capacity, medical supplies). Long-term forecasts quantify the human cost of unmitigated conflict.

### 4. In-Silico Experimentation

Simulate casualty dynamics under assumed parameters by setting tight priors and sampling from the prior predictive distribution. This allows:

- Sensitivity analysis: How do forecasts change with different parameter values?
- Study design: What sample sizes are needed to detect intervention effects?
- Validation: Can synthetic data be recovered accurately?

### 5. Humanitarian Planning

Combine estimates with resource models to optimize:

- Hospital bed allocation
- Medical supply distribution
- Evacuation corridor planning
- International aid targeting

## Implementation

**TRACE** is implemented in Python using:

- **[JAX](https://github.com/google/jax)**: For efficient, GPU-accelerated array operations and automatic differentiation
- **[NumPyro](https://num.pyro.ai)**: Probabilistic programming with the No-U-Turn Sampler (NUTS) {cite}`Hoffman2014` for Hamiltonian Monte Carlo inference
- **[ArviZ](https://python.arviz.org)**: For Bayesian model diagnostics and visualization

This stack provides:

- **Performance**: JAX's JIT compilation and GPU support enable rapid inference even on large datasets
- **Flexibility**: NumPyro's modeling language allows easy specification of complex hierarchical models
- **Diagnostics**: ArviZ provides comprehensive tools for assessing convergence and model fit

## Data Sources

**TRACE** integrates multiple data streams:

1. **Conflict Events**: ACLED provides geo-coded conflict event data with dates, locations, and reported fatalities
2. **Hospital Admissions**: Daily or weekly counts of conflict-related injuries at healthcare facilities
3. **Mortality Data**: National or regional death counts (e.g., from Ministry of Health reports)

For the Gaza conflict, we demonstrate using:

- **ACLED**: 27,815 conflict events (Sept 2023 - Dec 2024)
- **Tech for Palestine**: Daily mortality data from Gaza Ministry of Health
- **Hospital Data**: Currently simulated (methodology ready for real data when available)

## Mathematical Framework

The model has three key components:

### Observation Model

Hospital injuries and deaths are modeled as Poisson random variables:

$$
\begin{align}
H_{t,h} &\sim \text{Poisson}(\lambda_{t,h}) \\
D_t &\sim \text{Poisson}(\delta_t)
\end{align}
$$

where $H_{t,h}$ are injuries at hospital $h$ on day $t$, and $D_t$ are national deaths.

### Spatial Allocation

Expected injuries at each hospital depend on conflict events and spatial proximity:

$$
\lambda_{t,h} = \mu_w \sum_{e \in E_t} w_{e,h}
$$

where $E_t$ is the set of events on day $t$, $\mu_w$ is average wounded per event, and $w_{e,h}$ is the spatial weight:

$$
w_{e,h} = \frac{\exp(-d_{e,h}/\ell)}{\sum_{h'} \exp(-d_{e,h'}/\ell)}
$$

### Temporal Delays

Expected deaths combine immediate and delayed components:

$$
\delta_t = \mu_i \cdot |E_t| + p_{\text{late}} \sum_{k=1}^L I_{t-k} f_k
$$

where $\mu_i$ is immediate deaths per event, $I_t$ is total injuries on day $t$, $p_{\text{late}}$ is hospital fatality rate, and $f_k$ is the delay distribution.

## Prior Specifications

Default weakly informative priors:

- $\mu_w \sim \text{Exponential}(0.3)$ — mean of 3.3 wounded per event
- $\mu_i \sim \text{Exponential}(0.3)$ — mean of 3.3 immediate deaths per event
- $p_{\text{late}} \sim \text{Beta}(2, 10)$ — mean hospital fatality rate of ~17%
- $\ell \sim \text{Exponential}(1.0)$ — mean spatial scale of 1 degree

These priors are informed by literature on conflict casualties and can be adjusted based on context.

## Extensions

The basic model can be extended in several ways:

1. **Time-Varying Parameters**: Allow casualty rates to change over time (e.g., due to weapon changes, medical improvements)
2. **Covariates**: Include intervention indicators, conflict type, or other predictors
3. **Hierarchical Structure**: Partial pooling across multiple regions or conflict zones
4. **Overdispersion**: Negative Binomial distributions to handle extra-Poisson variation
5. **Missing Data**: Impute missing observations using the model structure

See the [Model Extensions](05_extensions.md) vignette for details.

## Getting Started

To begin using **TRACE**:

1. [Installation Instructions](../installation.md)
2. [Basic Example Tutorial](../tutorials/01_basic_example.md) — Simulated data walkthrough
3. [Gaza Analysis Tutorial](../tutorials/02_gaza_analysis.md) — Real data application
4. [Model Description](02_description.md) — Detailed mathematical specification

## References

```{bibliography}
:filter: docname in docnames
```

## Related Packages

**TRACE** draws inspiration from several related packages:

- **[epidemia](https://imperialcollegelondon.github.io/epidemia/)** (R): Semi-mechanistic epidemic models {cite}`Scott2020`
- **[EpiNow2](https://epiforecasts.io/EpiNow2/)** (R): Real-time epidemic forecasting
- **[PyMC](https://www.pymc.io/)**: General Bayesian modeling in Python
- **[NumPyro](https://num.pyro.ai)**: Probabilistic programming with JAX

---

*This vignette is part of the **TRACE** documentation. For questions or suggestions, please [open an issue](https://github.com/imperial-college/trace/issues).*
