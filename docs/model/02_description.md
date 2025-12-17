# Model Description

This vignette provides a detailed mathematical specification of the **TRACE** casualty model. We describe the basic model for a single region, then discuss extensions for multiple populations and time-varying effects.

## 1. Basic Model

We formulate the model for a single homogeneous population over $T$ days. The model integrates three data streams:

1. **Conflict Events**: $E_t$ events on day $t$, with locations $(x_e, y_e)$ for $e \in E_t$
2. **Hospital Injuries**: $H = (H_{1,1}, \ldots, H_{T,J})$ where $H_{t,j}$ is injuries at hospital $j$ on day $t$
3. **Deaths**: $D = (D_1, \ldots, D_T)$ where $D_t$ is national deaths on day $t$

### 1.1 Observation Models

In the simplest formulation, injuries and deaths can be modeled as Poisson random variables:

$$
\begin{align}
H_{t,j} &\sim \text{Poisson}(\lambda_{t,j}), \quad j = 1, \ldots, J \tag{1.1} \\
D_t &\sim \text{Poisson}(\delta_t), \quad t = 1, \ldots, T \tag{1.2}
\end{align}
$$

where $\lambda_{t,j}$ is the expected number of injuries at hospital $j$ on day $t$, and $\delta_t$ is the expected number of deaths on day $t$.

**Note on Overdispersion**: The Poisson assumption can be relaxed to accommodate extra-Poisson variation by using Negative Binomial distributions:

$$
\begin{align}
H_{t,j} &\sim \text{NegativeBinomial}(\lambda_{t,j}, \phi_H) \\
D_t &\sim \text{NegativeBinomial}(\delta_t, \phi_D)
\end{align}
$$

where $\phi_H$ and $\phi_D$ are overdispersion parameters. This extension is described in Section 1.5.

**Implementation Note**: The current TRACE implementation uses overdispersed likelihoods by default via a Gamma-Poisson mixture (NumPyro's `GammaPoisson`), which corresponds to a Negative Binomial model and reduces to a Poisson model as overdispersion goes to zero.

### 1.2 Hospital Injury Model

Expected injuries at each hospital are determined by:

1. The number and location of conflict events
2. Average casualties per event
3. Spatial allocation to hospitals

Formally:

$$
\lambda_{t,j} = \mu_w \sum_{e \in E_t} w_{e,j} \tag{1.3}
$$

where:

- $\mu_w > 0$ is the average number of wounded (hospitalized) per event
- $w_{e,j} \in [0,1]$ is the fraction of casualties from event $e$ allocated to hospital $j$
- $\sum_j w_{e,j} = 1$ for all events $e$

#### 1.2.1 Spatial Kernel

The spatial weights are determined by an exponential decay kernel:

$$
w_{e,j} = \frac{\exp(-d_{e,j}/\ell)}{\sum_{j'=1}^J \exp(-d_{e,j'}/\ell)} \tag{1.4}
$$

where:

- $d_{e,j}$ is the Euclidean distance between event $e$ and hospital $j$
- $\ell > 0$ is a spatial length scale parameter controlling decay

**Properties**:

- Hospitals closer to events receive higher weight
- As $\ell \to 0$, all weight goes to nearest hospital
- As $\ell \to \infty$, weights approach uniform distribution $w_{e,j} \to 1/J$
- The kernel respects the constraint $\sum_j w_{e,j} = 1$

**Alternative Kernels**: Other distance-based kernels can be used, such as:

- Power law: $w_{e,j} \propto d_{e,j}^{-\alpha}$
- Gaussian: $w_{e,j} \propto \exp(-d_{e,j}^2 / 2\ell^2)$
- Gravity model: $w_{e,j} \propto C_j / d_{e,j}^\alpha$ where $C_j$ is hospital capacity

### 1.3 Death Model

Deaths arise from two sources:

1. **Immediate Deaths**: Fatalities occurring at the conflict event location
2. **Delayed Deaths**: Hospital fatalities occurring days after injury

The total expected deaths on day $t$ is:

$$
\delta_t = \underbrace{\mu_i |E_t|}_{\text{immediate}} + \underbrace{p_{\text{late}} \sum_{k=1}^L I_{t-k} f_k}_{\text{delayed}} \tag{1.5}
$$

where:

- $\mu_i \geq 0$ is average immediate deaths per event
- $|E_t|$ is the number of events on day $t$
- $I_t = \sum_j H_{t,j}$ is total injuries on day $t$
- $p_{\text{late}} \in [0,1]$ is probability an injured person eventually dies
- $f = (f_1, \ldots, f_L)$ is the delay distribution with $\sum_k f_k = 1$
- $L$ is the maximum delay (support of delay distribution)

#### 1.3.1 Delay Distribution

The delay distribution $f_k$ represents the probability that death occurs $k$ days after injury. This can be:

**Empirically Estimated**: From medical literature on trauma outcomes. For conflict settings, typical estimates suggest:

- $f_1 \approx 0.5$ — 50% die within 1 day
- $f_2 \approx 0.3$ — 30% die on day 2
- $f_3 \approx 0.15$ — 15% die on day 3
- $f_4 \approx 0.05$ — 5% die on day 4+

**Parametric Form**: E.g., geometric or discretized exponential:

$$
f_k = (1-\rho)\rho^{k-1}, \quad k = 1, 2, \ldots
$$

**Flexible**: Estimated as free parameters with Dirichlet prior:

$$
f \sim \text{Dirichlet}(\alpha_1, \ldots, \alpha_L)
$$

#### 1.3.2 Interpretation

The convolution term $\sum_{k=1}^L I_{t-k} f_k$ represents the expected number of injuries from previous days that result in death on day $t$. Multiplying by $p_{\text{late}}$ scales this to the actual hospital fatality rate.

**Identifiability**: The parameters $p_{\text{late}}$ and $f$ are partially confounded — similar data can be explained by:

- High $p_{\text{late}}$ with short delays (most deaths occur quickly)
- Low $p_{\text{late}}$ with long delays (deaths are rare but spread over time)

Informative priors or fixed $f$ based on medical literature help resolve this.

## 2. Regression Structure

The model can be extended to include time-varying effects through regression:

### 2.1 Time-Varying Casualty Rates

Allow parameters to depend on covariates $X_t$:

$$
\begin{align}
\mu_{w,t} &= \mu_w \cdot \exp(X_t \beta_w) \\
\mu_{i,t} &= \mu_i \cdot \exp(X_t \beta_i)
\end{align}
$$

where $\beta_w$ and $\beta_i$ are regression coefficients.

**Example Covariates**:

- Ceasefire indicator: $X_t = \mathbb{1}(\text{ceasefire on day } t)$
- Conflict type: Different coefficients for different event types
- Time trend: $X_t = t$ to capture systematic changes

**Priors**: Coefficients typically receive normal priors:

$$
\beta_k \sim \mathcal{N}(0, \sigma_\beta^2)
$$

### 2.2 Autoreg

ressive Structure

For smoothly varying parameters, use random walk or autoregressive priors:

$$
\log \mu_{w,t} = \log \mu_{w,t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

This allows casualty rates to evolve over time without requiring explicit covariates.

**Implementation Note**: TRACE provides an opt-in random-walk model for $\mu_{w,t}$ and $\mu_{i,t}$ via `trace.model.casualty_model_random_walk`.

## 3. Parameter Priors

Default weakly informative priors are chosen to be:

1. Proper (for computational stability)
2. Regularizing (preventing extreme values)
3. Scientifically plausible

### 3.1 Casualty Rate Priors

$$
\begin{align}
\mu_w &\sim \text{Exponential}(\lambda_w) \\
\mu_i &\sim \text{Exponential}(\lambda_i)
\end{align}
$$

With default rates $\lambda_w = \lambda_i = 0.3$, giving prior means around 3.3 casualties per event.

**Rationale**: Exponential priors are:

- Non-negative (as required)
- Weakly informative (allowing a wide range)
- Proper (integrable)

**Sensitivity**: Users should assess sensitivity to these choices, particularly if prior knowledge suggests very different casualty rates.

### 3.2 Hospital Fatality Rate

$$
p_{\text{late}} \sim \text{Beta}(\alpha, \beta)
$$

Default: $\alpha = 2, \beta = 10$, giving:

- Prior mean: $\mathbb{E}[p_{\text{late}}] = \alpha/(\alpha + \beta) \approx 0.17$
- Prior mode: $(\\alpha - 1)/(\alpha + \beta - 2) \approx 0.1$
- Support: $(0, 1)$ as required for a probability

**Rationale**: Based on medical literature suggesting 10-20% hospital fatality rates for severe trauma in conflict settings {cite}`GlobalBurdenOfDisease2020`.

### 3.3 Spatial Length Scale

$$
\ell \sim \text{Exponential}(\lambda_\ell)
$$

Default: $\lambda_\ell = 1.0$, giving prior mean of 1 degree (approximately 100 km at mid-latitudes).

**Interpretation**:

- Small $\ell$ (e.g., 0.1 degree): Very local allocation, casualties go to nearest hospital
- Medium $\ell$ (e.g., 1 degree): Regional allocation, distance matters but not exclusively
- Large $\ell$ (e.g., 10 degrees): Diffuse allocation, distance less important

**Context-Specific**: For small regions like Gaza ($\sim$40 km long), $\ell \approx 0.5$ degrees is reasonable. For larger countries, $\ell$ may be several degrees.

## 4. Posterior Inference

The joint posterior distribution is:

$$
p(\mu_w, \mu_i, p_{\text{late}}, \ell, \phi_H, \phi_D | H, D, E) \propto p(\mu_w) p(\mu_i) p(p_{\text{late}}) p(\ell) p(\phi_H) p(\phi_D) \times \mathcal{L}(H | \mu_w, \ell, \phi_H, E) \times \mathcal{L}(D | \mu_i, p_{\text{late}}, \phi_D, H, E)
$$

where:

$$
\begin{align}
\mathcal{L}(H | \mu_w, \ell, \phi_H, E) &= \prod_{t=1}^T \prod_{j=1}^J \text{NegBin}(H_{t,j} | \lambda_{t,j}, \phi_H) \\
\mathcal{L}(D | \mu_i, p_{\text{late}}, \phi_D, H, E) &= \prod_{t=1}^T \text{NegBin}(D_t | \delta_t, \phi_D)
\end{align}
$$

This posterior is represented in a NumPyro model and sampled using the No-U-Turn Sampler (NUTS) {cite}`Hoffman2014`, an adaptive Hamiltonian Monte Carlo algorithm.

### 4.1 MCMC Diagnostics

After sampling, convergence should be assessed using:

1. **Trace Plots**: Visual inspection of parameter traces
2. **$\hat{R}$ Statistic** {cite}`Gelman2013`: Should be $< 1.01$ for all parameters
3. **Effective Sample Size** (ESS): Should be $> 100$ per chain
4. **Divergences**: Should be zero or very few

See the {ref}`convergence-diagnostics` for details.

## 5. Posterior Predictive Distribution

For model checking, we sample from the posterior predictive distribution:

$$
\begin{align}
\tilde{H}_{t,j} &\sim \text{NegBin}(\lambda_{t,j}^{(s)}, \phi_H^{(s)}) \\
\tilde{D}_t &\sim \text{NegBin}(\delta_t^{(s)}, \phi_D^{(s)})
\end{align}
$$

for each posterior sample $s = 1, \ldots, S$.

**Posterior Predictive Checks**: Compare observed data $H, D$ to replicated data $\tilde{H}, \tilde{D}$:

- **Coverage**: Do 95% credible intervals contain 95% of observations?
- **Mean**: Is $\mathbb{E}[\tilde{H}] \approx H$?
- **Dispersion**: Is $\text{Var}(\tilde{H}) \approx \text{Var}(H)$?

Systematic discrepancies suggest model misspecification.

## 6. Forecasting

To forecast $h$ days into the future, we need:

1. **Scenario for Future Events**: $\tilde{E}_{T+1}, \ldots, \tilde{E}_{T+h}$
2. **Posterior Samples**: $\theta^{(1)}, \ldots, \theta^{(S)}$ where $\theta = (\mu_w, \mu_i, p_{\text{late}}, \ell)$

For each posterior sample $s$:

1. Simulate injuries: $\tilde{H}_{t,j}^{(s)} \sim \text{NegBin}(\mu_w^{(s)} \sum_{e \in \tilde{E}_t} w_{e,j}^{(s)}, \phi_H^{(s)})$
2. Simulate deaths: $\tilde{D}_t^{(s)} \sim \text{NegBin}(\mu_i^{(s)} |\tilde{E}_t| + p_{\text{late}}^{(s)} \sum_k \tilde{I}_{t-k}^{(s)} f_k, \phi_D^{(s)})$

Aggregate across samples to obtain forecast distributions.

**Scenarios**:

- **Baseline**: $\tilde{E}_t$ matches recent historical rate
- **Ceasefire**: $|\tilde{E}_t| \to 0$
- **Escalation**: $|\tilde{E}_t|$ increases by factor $\alpha > 1$
- **Custom**: User-specified event counts

## References

```{bibliography}
:filter: docname in docnames
```

---

*See [Model Implementation](03_implementation.md) for details on the NumPyro code.*
