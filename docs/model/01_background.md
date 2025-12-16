# Background

The open-source Python package **TRACE** (Temporal and Regional Analysis of Conflict Events) provides a framework for Bayesian, regression-oriented modeling of the temporal dynamics of conflict casualties. Typically these models are fit to multiple data streams—conflict event data, hospital admission records, and mortality time series—for a given population and period. Casualty dynamics are described explicitly through a semi-mechanistic process; observed data are linked to latent conflict intensity, which is in turn modeled through discrete event processes tempered by spatially-varying allocation weights. Regression models are specified for several objects in the model. For example, casualty rates can be expressed as transformed predictors, which may include both covariates and autoregressive terms. A range of prior distributions can be specified for unknown parameters by leveraging functionality from **NumPyro** {cite}`Phan2019` and **JAX** {cite}`Bradbury2018`. Multilevel models are supported by partially pooling covariate effects appearing in predictors between multiple populations or regions.

The mathematical framework motivating the implemented models builds upon semi-mechanistic approaches developed for epidemic modeling {cite}`Flaxman2020, Bhatt2020`. Specific analyses using such models have appeared during the COVID-19 pandemic and have been used to estimate the effect of control measures {cite}`Flaxman2020, Mellan_2020, Olney_2021`, and to forecast disease dynamics under assumed epidemiological parameters and mitigation scenarios {cite}`Vollmer_2020, Hawryluk_2020`. The modeling approach has been extended to estimate differences in transmissibility between COVID-19 lineages {cite}`Faria_2021, Volz_2021`. **TRACE** adapts these methods to the conflict domain, where discrete violent events replace continuous disease transmission, and spatial allocation to hospitals replaces age-stratified contact matrices.

Models of casualty dynamics are commonly classified as either mechanistic or statistical {cite}`Myers2000`. Mechanistic models derive casualty patterns from theoretical considerations over how conflicts unfold, such as Lanchester's equations for combat attrition {cite}`Lanchester1914`, agent-based simulations of tactical engagements {cite}`Epstein2002`, or network models of insurgent activity {cite}`Bohorquez2009`. These models propose explicit equations or rules that govern the evolution of combatants and casualties over time. Such models are motivated by theories of military strategy, resource constraints, and behavioral dynamics. Purely statistical models, on the other hand, make few assumptions over the generative mechanism, and instead infer future dynamics from the history of the process and related covariates. Examples include Generalized Linear Models (GLMs) for event counts, time series approaches including Auto Regressive Integrated Moving Average (ARIMA) models {cite}`Box1970`, and more modern forecasting methods based on machine learning {cite}`Mueller2020, Blair2021`.

**TRACE** provides models which are *semi-mechanistic*. These are statistical models that explicitly describe casualty generation processes. Discrete conflict events precipitate casualties, which are allocated spatially to hospitals based on distance-weighted kernels. Injuries lead to deaths with explicit delay distributions, allowing temporal dynamics to be tracked explicitly. This approach has been used in multiple previous epidemic modeling works {cite}`Fraser2007, Cori2013, Nouvellet2018, Cauchemez2008` and has analogues in conflict modeling through spatial point process approaches {cite}`Zammit2014, Schutte2021`. In addition, population adjustments may be applied to account for displacement or changes in population at risk. The models are *statistical* in the sense that they define a likelihood function for the observed data. After also specifying prior distributions for model parameters, samples from the posterior can then be obtained using either Hamiltonian Monte Carlo or Variational Bayes methods.

The Bayesian approach has certain advantages in this context. Several aspects of these models are fundamentally unidentified {cite}`Roosa2019`. For most conflicts, casualty counts suffer from systematic under-reporting, with true tolls often substantially exceeding official figures {cite}`Spagat2009, Obermeyer2008`. Recorded counts could be explained by high casualties and low reporting, or alternatively by moderate casualties and better documentation. If a series of interventions are applied in sequence to reduce violence (ceasefires, humanitarian corridors, international pressure), then their effects may be confounded and difficult to disentangle {cite}`Bhatt2020`. Bayesian approaches using MCMC allow full exploration of posterior correlations between such coupled parameters. Informative, or weakly informative, priors may be incorporated to regularize, and help to mitigate identifiability problems, which may otherwise pose difficulties for sampling {cite}`Gelman2008, Gelman2013`.

**TRACE**'s functionality can be used for a number of purposes. A researcher can simulate casualty dynamics under assumed parameters by setting tight priors around the assumed values. It is then possible to sample directly from the prior distribution without conditioning on data. This allows *in-silico* experimentation; for example, to assess the effect of varying a single parameter (conflict intensity, hospital fatality rates, spatial accessibility). Another goal of modeling is to assess whether a simple and parsimonious model of reality can replicate observed phenomena. This helps to isolate processes helpful for explaining the data. Models of varying complexity can be specified within **TRACE**, largely as a result of its regression-oriented framework. Posterior predictive checks can be used to assess model fit. If the model is deemed misspecified, additional features may be considered. This could be modeling population displacement, explicit modeling of mass casualty events {cite}`Neumayer2021`, alternative and over-dispersed models for the data, or more flexible functional forms for casualty rates or hospital allocation patterns. This can be done rapidly within **TRACE**'s framework.

Forecasting models are critical during ongoing conflicts as they are used to inform humanitarian response and policy decisions under uncertainty. As a sign of their importance, organizations such as the Armed Conflict Location & Event Data Project (ACLED) {cite}`Raleigh2010` and the Uppsala Conflict Data Program (UCDP) {cite}`Pettersson2021` maintain extensive databases of conflict events to support analytical and forecasting efforts. Long-term forecasts quantify the human cost of unmitigated conflict and provide a baseline from which to infer the effects of interventions. Short-term forecasts are crucial in informing decisions on how to distribute resources such as medical supplies, field hospitals, or evacuation capacity, or whether humanitarian organizations should increase their presence in affected areas. Traditional statistical approaches often give unrealistic long-term forecasts as they do not explicitly account for the mechanisms driving casualties or the effects of spatial structure. The semi-mechanistic approach of **TRACE** combines the strengths of statistical approaches with explicit casualty generation dynamics and can thus be used for forecasting at different time horizons.

## The TRACE Framework

The **TRACE** package implements semi-mechanistic Bayesian models for conflict casualty analysis. The core framework consists of three inter-related components:

### 1. Conflict Event Process

Violent events are the primary drivers of casualties. These are typically recorded in event databases such as ACLED {cite}`Raleigh2010`, which document the date, location, type, and sometimes reported casualties of political violence events. Let $E_t$ denote the set of events occurring on day $t$, with $|E_t|$ denoting the count. Each event $e \in E_t$ is associated with a geographic location $(x_e, y_e)$ and potentially an event type (e.g., airstrike, shelling, armed clash).

Unlike epidemics where new infections arise endogenously from previous infections, conflict events are often driven by external factors such as military operations, political decisions, or escalation dynamics. However, some conflicts do exhibit self-exciting properties {cite}`Townsley2003, Mohler2011`, where past violence increases the probability of future violence. **TRACE** currently treats events as exogenous inputs but can be extended to model event generation endogenously.

### 2. Spatial Allocation to Hospitals

Casualties from conflict events seek treatment at hospitals. The allocation of casualties to specific hospitals depends on:

- **Geographic proximity**: Closer hospitals are more accessible
- **Hospital capacity**: Larger facilities may draw from wider catchment areas
- **Infrastructure**: Road networks, checkpoints, and damage affect accessibility
- **Security**: Active conflict zones may make certain routes impassable

**TRACE** models this through spatial kernel functions. For event $e$ and hospital $h$ separated by distance $d_{e,h}$, the proportion of casualties allocated to hospital $h$ is:

$$
w_{e,h} = \frac{k(d_{e,h}; \ell)}{\sum_{h'} k(d_{e,h'}; \ell)}
$$

where $k(\cdot; \ell)$ is a kernel function with length scale parameter $\ell$. The exponential kernel $k(d; \ell) = \exp(-d/\ell)$ is used by default, giving heavier weight to nearby facilities. Alternative kernels (Gaussian, power-law, gravity-model based) can be specified to capture different accessibility patterns {cite}`Gething2012}.

The spatial structure distinguishes **TRACE** from aggregate national-level models and allows for:

- Hospital-specific forecasting for resource planning
- Identifying underserved geographic areas
- Assessing the impact of hospital closures or damage
- Optimizing placement of mobile medical units

### 3. Temporal Dynamics: Injury to Death

Not all casualties die immediately. Many injured individuals are treated in hospitals, with outcomes (survival or death) occurring over subsequent days or weeks. This creates temporal delays between injury incidence and mortality observations. **TRACE** models this through:

$$
D_t = \underbrace{\mu_i |E_t|}_{\text{immediate deaths}} + \underbrace{p_{\text{late}} \sum_{k=1}^L I_{t-k} f_k}_{\text{delayed deaths}}
$$

where:

- $D_t$ is deaths on day $t$
- $\mu_i$ is average immediate deaths per event
- $I_{t-k}$ is injuries on day $t-k$
- $p_{\text{late}}$ is hospital fatality rate
- $f_k$ is delay distribution (probability death occurs $k$ days post-injury)

The delay distribution $f$ can be informed by medical literature on trauma outcomes {cite}`Champion2003, Eastridge2012`, which suggests that most conflict-related deaths among hospitalized patients occur within the first 72 hours, with a tail extending to weeks for severe injuries complicated by sepsis or organ failure.

This temporal structure allows **TRACE** to:

- Separate immediate battlefield deaths from hospital deaths
- Estimate hospital fatality rates (indicator of care quality and injury severity)
- Project future deaths from current injuries (even if conflict stops)
- Distinguish between high-lethality (many immediate deaths) and high-injury (many hospitalizations) scenarios

## Mathematical Foundations

The model is formulated as a hierarchical Bayesian model with three likelihood components:

### Hospital Injury Likelihood

Daily injuries at hospital $h$ are modeled as:

$$
H_{t,h} \sim \text{Poisson}(\lambda_{t,h})
$$

where the rate $\lambda_{t,h}$ depends on events and spatial allocation:

$$
\lambda_{t,h} = \mu_w \sum_{e \in E_t} w_{e,h}
$$

Here $\mu_w$ is the average number of wounded per event. This parameter varies substantially across conflict types: airstrikes on residential areas produce many casualties {cite}`Guha2015`, while targeted assassinations may produce few.

### Death Likelihood

National daily deaths are modeled as:

$$
D_t \sim \text{Poisson}(\delta_t)
$$

with rate $\delta_t$ given by the sum of immediate and delayed components as described above.

### Extensions

The basic Poisson framework can be extended to handle:

- **Overdispersion**: Negative Binomial distributions $\text{NB}(\mu, \phi)$ to capture extra-Poisson variation
- **Time-varying rates**: $\mu_{w,t} = \mu_w \exp(X_t \beta)$ where $X_t$ are covariates (ceasefire indicators, conflict intensity metrics)
- **Event-type specific effects**: Different $\mu_w$ and $\mu_i$ for airstrikes vs. ground combat
- **Hierarchical structure**: Partial pooling of parameters across multiple conflict zones

## Prior Specifications

Default priors are weakly informative, balancing between being proper (for computational stability) and allowing data to dominate:

- **Casualty rates**: $\mu_w, \mu_i \sim \text{Exponential}(0.3)$, implying prior mean of 3.3 casualties per event
- **Hospital fatality**: $p_{\text{late}} \sim \text{Beta}(2, 10)$, implying prior mean ~17%, consistent with trauma literature
- **Spatial scale**: $\ell \sim \text{Exponential}(1.0)$, implying prior mean of 1 degree (∼100 km)

These defaults can and should be adjusted based on:

- **Context-specific information**: Urban vs. rural, aerial bombardment vs. ground combat
- **Historical data**: Casualty rates from similar conflicts
- **Expert knowledge**: Input from humanitarian organizations or military analysts

Sensitivity to priors should always be assessed, particularly for parameters that are weakly identified by data {cite}`Gelman2008`.

## Inference

Posterior inference is performed using the No-U-Turn Sampler (NUTS) {cite}`Hoffman2014`, an adaptive variant of Hamiltonian Monte Carlo (HMC) that automatically tunes step sizes and trajectory lengths. NUTS has become the default inference algorithm in modern probabilistic programming languages due to its efficiency on high-dimensional, correlated posteriors.

**TRACE** uses **NumPyro** {cite}`Phan2019`, which implements NUTS on top of **JAX** {cite}`Bradbury2018`. This provides:

- **Performance**: JIT compilation and GPU acceleration enable rapid inference
- **Scalability**: Can handle datasets with hundreds of thousands of events
- **Diagnostics**: Automatic computation of $\hat{R}$, effective sample size, and divergence warnings

Users should always check convergence diagnostics {cite}`Gelman2013`:

- **$\hat{R} < 1.01$**: Potential scale reduction factor, indicates chains have converged
- **ESS > 100 per chain**: Effective sample size, indicates sufficient posterior exploration
- **Zero divergences**: Indicates sampler is exploring posterior geometry correctly

## Comparison to Epidemic Models

The **TRACE** framework draws heavily from semi-mechanistic epidemic models, particularly the approach of Flaxman et al. {cite}`Flaxman2020` and the **epidemia** R package {cite}`Scott2020`. However, there are key differences:

| Aspect | Epidemic Models | Conflict Models (TRACE) |
|--------|----------------|------------------------|
| **Generative Process** | Infections → infections (self-exciting) | Events → casualties (exogenous events) |
| **Temporal Structure** | Renewal equation with generation time | Discrete events with injury-to-death delays |
| **Spatial Structure** | Age-stratified contact matrices | Geographic kernel-based hospital allocation |
| **Data Streams** | Cases, deaths, seroprevalence | Events, hospital admissions, mortality |
| **Interventions** | NPIs (lockdowns, masks, vaccines) | Ceasefires, humanitarian corridors |
| **Identifiability** | $R_t$ vs. ascertainment confounded | Immediate vs. delayed deaths confounded |

Despite differences, many methodological innovations transfer:

- Hierarchical regression for $R_t$ → Hierarchical regression for casualty rates
- Partial pooling across countries → Partial pooling across conflict zones
- Posterior predictive forecasting → Scenario-based casualty projections

## Comparison to Traditional Conflict Models

**TRACE** differs from traditional conflict forecasting approaches:

### Lanchester Models

Lanchester's attrition equations {cite}`Lanchester1914` model combat as deterministic differential equations governing force sizes:

$$
\frac{dN_1}{dt} = -\alpha_2 N_2, \quad \frac{dN_2}{dt} = -\alpha_1 N_1
$$

where $N_i$ is the size of force $i$ and $\alpha_i$ is its lethality. These are highly interpretable but:

- Require knowing force sizes (often unavailable in civil conflicts)
- Assume closed-system, symmetric warfare (unrealistic for asymmetric conflicts)
- No spatial structure or stochasticity
- Difficult to fit to observational data

Recent applications to Gaza {cite}`Kress2025` demonstrate their utility but also limitations.

### Agent-Based Models

ABMs {cite}`Epstein2002` simulate individual combatants and civilians, allowing rich spatial and behavioral dynamics. However:

- Computationally intensive, limiting use for real-time forecasting
- Many free parameters, difficult to calibrate to data
- Validation challenging due to complexity
- Less suitable for probabilistic uncertainty quantification

### Statistical Forecasting

Time series models (ARIMA, exponential smoothing) or machine learning (random forests, neural networks) can provide good short-term forecasts {cite}`Mueller2020, Blair2021` but:

- Lack mechanistic interpretation (black-box)
- Poor extrapolation beyond training regime
- No explicit spatial structure
- Difficult to incorporate scenario assumptions (e.g., ceasefire effects)

**TRACE** occupies a middle ground: more mechanistic than pure time series approaches, more statistically grounded than deterministic combat models, more computationally tractable than detailed ABMs.

## Data Requirements and Sources

**TRACE** requires three primary data streams:

### 1. Conflict Events

Geocoded event data with dates and locations. Sources include:

- **ACLED** {cite}`Raleigh2010`: Global coverage, updated weekly, includes event types and reported fatalities
- **UCDP GED** {cite}`Sundberg2013`: Similar coverage, slightly different methodology
- **Local databases**: News scraping, social media monitoring, witness reports

Events should minimally include date and coordinates. Additional covariates (event type, weapon system, actors) enable richer models.

### 2. Hospital Admissions

Daily or weekly counts of conflict-related injuries at healthcare facilities. This is often the most challenging data to obtain:

- **Official health systems**: Ministry of Health reports (often aggregated, may be incomplete)
- **Humanitarian organizations**: MSF, ICRC, WHO often collect facility-level data but may not publicly release
- **Sentinel surveillance**: Detailed data from subset of facilities

When unavailable, **TRACE** can be fit to national mortality data alone, though this reduces inferential power.

### 3. Mortality Data

National or regional death counts. Sources include:

- **Official reports**: Ministry of Health daily/weekly bulletins
- **Surveillance systems**: Civil registration, burial records
- **Alternative sources**: Media monitoring, NGO documentation (e.g., Tech for Palestine for Gaza)

Mortality data is typically more complete than injury data but still suffers from under-reporting, particularly during intense conflict when recording systems break down {cite}`Obermeyer2008`.

## Case Study: Gaza (2023-2024)

**TRACE** has been applied to the Gaza conflict following the October 7, 2023 escalation. This demonstrates the framework's utility in an active humanitarian crisis:

- **Events**: 27,815 ACLED events (Sept 2023 - Dec 2024)
- **Mortality**: Daily reports from Gaza Ministry of Health via Tech for Palestine
- **Findings**: 
  - ~5 hospital admissions per conflict event
  - ~0.6 immediate deaths per event
  - ~49% hospital fatality rate (extremely high, indicating overwhelmed medical system)
  - Under baseline scenario: ~3,900 deaths projected over 30 days
  - Under ceasefire: ~400 deaths (mostly delayed from existing injuries)
  - **Potential lives saved: 3,500 over 30 days**

These findings inform humanitarian planning and international advocacy. See the [Gaza Analysis Tutorial](../tutorials/02_gaza_analysis.md) for full details.

## Limitations and Caveats

Users should be aware of several important limitations:

### 1. Data Quality

All inferences are conditional on data quality. Under-reporting, selective reporting, and measurement error can bias estimates. Sensitivity analyses varying assumptions about reporting rates are essential.

### 2. Identifiability

Some parameters are only weakly identified. For example:

- **Immediate vs. delayed deaths**: If delay distribution is misspecified, $\mu_i$ and $p_{\text{late}}$ can trade off
- **Spatial allocation**: Without hospital-level data, spatial kernel parameters are unidentified
- **Reporting rates**: Casualty rates and ascertainment are confounded

Informative priors and external validation data (e.g., satellite imagery, media reports) can help.

### 3. Causal Inference

**TRACE** is primarily a forecasting and descriptive tool. Causal claims about intervention effects require careful design:

- Adjust for confounders (conflict intensity, seasonality)
- Use difference-in-differences or interrupted time series designs
- Account for anticipation effects (if ceasefire is anticipated, behavior changes before)

### 4. Indirect Deaths

**TRACE** models only direct conflict casualties. Indirect deaths from disease, malnutrition, and healthcare system collapse often exceed direct deaths {cite}`Burnham2006, Roberts2013` but require different modeling approaches (e.g., demographic models, epidemiological simulations).

### 5. Ethical Considerations

Casualty modeling raises ethical concerns:

- **Victim dignity**: Numbers should not obscure individual suffering
- **Political use**: Models may be used selectively to support political narratives
- **Dual use**: Methods could potentially inform targeting decisions

Researchers should engage with humanitarian organizations, adhere to ethical guidelines {cite}`Zwitter2019`, and communicate uncertainty transparently.

## Software Implementation

**TRACE** is implemented in Python using:

- **JAX** {cite}`Bradbury2018`: Array operations, automatic differentiation, GPU support
- **NumPyro** {cite}`Phan2019`: Probabilistic programming with NUTS inference
- **ArviZ** {cite}`Kumar2019`: Bayesian diagnostics and visualization
- **Pandas/NumPy**: Data manipulation

The package is open-source (MIT license), version-controlled on GitHub, and includes:

- Comprehensive documentation (this website)
- Worked examples with real data
- Unit tests for core functionality
- Continuous integration

## Related Work and Packages

**TRACE** builds on and complements:

- **epidemia** (R) {cite}`Scott2020`: Semi-mechanistic epidemic models (inspiration for TRACE)
- **EpiNow2** (R) {cite}`Abbott2020`: Real-time epidemic forecasting
- **PyMC** (Python) {cite}`Salvatier2016`: General Bayesian modeling
- **Stan** (C++) {cite}`Carpenter2017`: Probabilistic programming with HMC

Conflict-specific tools:

- **conflict_forecast** (Python) {cite}`Hegre2021`: Structural conflict prediction models
- **icews** (R): Political event data analysis
- **ViEWS** (Python) {cite}`Hegre2019`: Violence early warning system

**TRACE** is unique in combining:

- Semi-mechanistic casualty generation
- Spatial hospital allocation
- Bayesian uncertainty quantification
- Modern probabilistic programming tools (NumPyro/JAX)

## Getting Started

To begin using **TRACE**:

1. **Installation**: See [Installation Guide](../installation.md)
2. **Basic Tutorial**: Work through [simulated data example](../tutorials/01_basic_example.md)
3. **Real Data**: Study [Gaza analysis](../tutorials/02_gaza_analysis.md)
4. **Model Details**: Read [mathematical description](02_description.md)
5. **Implementation**: Explore [NumPyro code](03_implementation.md)

## Contributing

**TRACE** is under active development. Contributions are welcome:

- **Bug reports**: [GitHub Issues](https://github.com/OJWatson/trace/issues)
- **Feature requests**: Discuss via Issues
- **Code contributions**: Submit Pull Requests
- **Documentation**: Corrections and clarifications appreciated

See [Contributing Guide](../contributing.md) for details.

## Acknowledgments

**TRACE** development was supported by Imperial College London. The modeling framework draws heavily from epidemic models developed by the MRC Centre for Global Infectious Disease Analysis, particularly the **epidemia** package {cite}`Scott2020`. Data from ACLED {cite}`Raleigh2010` and Tech for Palestine made the Gaza analysis possible.

## References

```{bibliography}
:filter: docname in docnames
```

---

*This background vignette provides context for the **TRACE** package. For mathematical details, see [Model Description](02_description.md). For usage instructions, see [Tutorials](../tutorials/01_basic_example.md).*
