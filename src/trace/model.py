"""
Bayesian hierarchical model for conflict casualties using NumPyro.

This module implements the core probabilistic model that integrates conflict events,
hospital admissions, and mortality data with spatial and temporal dynamics.
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


def spatial_kernel_weights(
    event_coords: jnp.ndarray, hospital_coords: jnp.ndarray, ell: float
) -> jnp.ndarray:
    """
    Compute normalized spatial kernel weights for event-hospital allocation.

    Uses an exponential decay kernel to model how casualties from a conflict event
    are distributed among hospitals based on distance. Closer hospitals receive
    more casualties.

    Parameters
    ----------
    event_coords : jnp.ndarray
        Array of shape (E, 2) with (latitude, longitude) for each event
    hospital_coords : jnp.ndarray
        Array of shape (H, 2) with (latitude, longitude) for each hospital
    ell : float
        Length scale parameter controlling spatial decay (in coordinate units)

    Returns
    -------
    jnp.ndarray
        Array of shape (E, H) where each row sums to 1, representing the
        fraction of casualties from each event allocated to each hospital

    Notes
    -----
    The kernel uses exponential decay: weight ∝ exp(-distance/ell)
    For small regions, Euclidean distance on lat/lon is acceptable.
    For larger regions, consider using haversine distance.
    """
    # Compute pairwise distances between events and hospitals
    # event_coords: (E, 2), hospital_coords: (H, 2)
    # Broadcasting: (E, 1, 2) - (1, H, 2) = (E, H, 2)
    dists = jnp.linalg.norm(
        event_coords[:, None, :] - hospital_coords[None, :, :], axis=2
    )  # shape (E, H)

    # Compute unnormalized weights using exponential decay kernel
    weights = jnp.exp(-dists / ell)

    # Normalize weights for each event so they sum to 1 across hospitals
    weight_sums = jnp.sum(weights, axis=1, keepdims=True) + 1e-8  # avoid division by zero
    norm_weights = weights / weight_sums

    return norm_weights  # shape (E, H)


def casualty_model(
    events_by_day: np.ndarray,
    event_day_index: np.ndarray,
    event_coords: np.ndarray,
    hospital_coords: np.ndarray,
    injuries_obs: np.ndarray,
    deaths_obs: np.ndarray,
    delay_probs: Optional[np.ndarray] = None,
) -> None:
    """
    Bayesian hierarchical model for conflict casualties.

    This model integrates three data streams:
    1. ACLED conflict events (exogenous input)
    2. Hospital casualty admissions (observed)
    3. National mortality data (observed)

    The model uses spatial kernels to allocate casualties to hospitals and
    delay distributions to model the time from injury to death.

    Parameters
    ----------
    events_by_day : np.ndarray
        1D array of length T with count of events per day
    event_day_index : np.ndarray
        1D array of length E listing the day index (0..T-1) of each event
    event_coords : np.ndarray
        Array of shape (E, 2) with (lat, lon) for each event
    hospital_coords : np.ndarray
        Array of shape (H, 2) with (lat, lon) for each hospital
    injuries_obs : np.ndarray
        Array of shape (T, H) with observed daily injured counts at each hospital
    deaths_obs : np.ndarray
        1D array of length T with observed daily death counts (national)
    delay_probs : np.ndarray, optional
        1D array for distribution of delay from injury to death.
        If None, uses default [0.5, 0.3, 0.15, 0.05] for 1-4 day delays

    Returns
    -------
    None
        This is a NumPyro model function that defines the probabilistic model.
        It samples from priors and conditions on observed data.

    Model Parameters (Inferred)
    ---------------------------
    mu_w : float
        Average number of wounded (hospitalized) per conflict event
    mu_i : float
        Average number of immediate fatalities per event
    p_late : float
        Probability an injured person eventually dies (hospital fatality rate)
    ell : float
        Spatial length scale for hospital allocation kernel

    Notes
    -----
    The model assumes:
    - Casualties from events are allocated to hospitals via spatial kernel
    - Immediate deaths occur on the event day
    - Injured individuals may die later according to delay distribution
    - Observations use overdispersed count likelihoods (Negative Binomial via Gamma-Poisson)

    Examples
    --------
    This function is used with NumPyro's MCMC inference:

    >>> from numpyro.infer import MCMC, NUTS
    >>> kernel = NUTS(casualty_model)
    >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
    >>> mcmc.run(rng_key, events_by_day, event_day_index, ...)
    """
    n_days = len(events_by_day)
    n_hospitals = hospital_coords.shape[0]

    if delay_probs is None:
        # Default delay distribution: 50% die after 1 day, 30% after 2 days, etc.
        delay_probs = jnp.array([0.5, 0.3, 0.15, 0.05])
        delay_probs = delay_probs / jnp.sum(delay_probs)  # ensure normalized

    # ========== Priors for unknown parameters ==========

    # Average wounded per event (mean ~3.3 with Exponential(0.3))
    mu_w = numpyro.sample("mu_w", dist.Exponential(0.3))

    # Average immediate deaths per event
    mu_i = numpyro.sample("mu_i", dist.Exponential(0.3))

    # Probability an injured person dies later (hospital fatality rate)
    # Beta(2, 10) has mean ~0.17, reflecting that most injuries don't result in death
    p_late = numpyro.sample("p_late", dist.Beta(2, 10))

    # Spatial length scale for hospital allocation (in coordinate units)
    # Exponential(1.0) gives mean of 1.0 degree
    ell = numpyro.sample("ell", dist.Exponential(1.0))

    phi_hosp = numpyro.sample("phi_hosp", dist.Exponential(1.0))
    phi_death = numpyro.sample("phi_death", dist.Exponential(1.0))

    # ========== Spatial allocation of casualties to hospitals ==========

    # Compute spatial weights for each event-hospital pair
    norm_weights = spatial_kernel_weights(jnp.array(event_coords), jnp.array(hospital_coords), ell)

    # Accumulate event contributions to each hospital per day
    # effective_events[d,h] = sum of normalized weights of all events on day d going to hospital h
    effective_events = jnp.zeros((n_days, n_hospitals))
    effective_events = effective_events.at[event_day_index].add(norm_weights)

    # ========== Hospital injuries likelihood ==========

    # Expected injuries at each hospital per day
    # lambda_injuries[d,h] = mu_w * effective_events[d,h]
    lam_injuries = mu_w * effective_events  # shape (T, H)

    # Ensure positive rates (add small epsilon for numerical stability)
    lam_injuries = jnp.maximum(lam_injuries, 1e-8)

    # Observe hospital injuries via an overdispersed Gamma-Poisson likelihood
    obs_inj = numpyro.sample(
        "obs_injuries",
        dist.GammaPoisson(concentration=phi_hosp, rate=phi_hosp / lam_injuries).to_event(2),
        obs=injuries_obs,
    )

    # ========== Delayed deaths from injuries ==========

    # Total injuries per day (summing across hospitals)
    # Use sampled injuries if obs was None, otherwise use provided injuries
    injuries_to_use = obs_inj if injuries_obs is None else jnp.array(injuries_obs)
    injuries_total = jnp.nansum(injuries_to_use, axis=1)  # shape (T,)

    # Convolve injuries with delay distribution to get expected delayed deaths
    delay_len = delay_probs.shape[0]  # length of delay distribution support
    pad = jnp.pad(injuries_total, (0, delay_len))  # pad for convolution

    # Compute convolution: deaths on day t from injuries on previous days
    conv_deaths = jnp.zeros(n_days)
    for k in range(delay_len):
        # Deaths occurring k+1 days after injury
        conv_deaths = conv_deaths + pad[k : n_days + k] * delay_probs[k]

    # Expected late deaths = p_late * convolved injuries
    expected_late_deaths = p_late * conv_deaths

    # ========== Immediate deaths from events ==========

    events_by_day_arr = jnp.array(events_by_day)
    expected_immediate_deaths = mu_i * events_by_day_arr

    # ========== Total deaths likelihood ==========

    # Total expected deaths = immediate + delayed
    expected_deaths = expected_immediate_deaths + expected_late_deaths

    # Ensure positive rates (add small epsilon for numerical stability)
    expected_deaths = jnp.maximum(expected_deaths, 1e-8)

    # Observe national deaths via an overdispersed Gamma-Poisson likelihood
    numpyro.sample(
        "obs_deaths",
        dist.GammaPoisson(concentration=phi_death, rate=phi_death / expected_deaths),
        obs=deaths_obs,
    )


def casualty_model_random_walk(
    events_by_day: np.ndarray,
    event_day_index: np.ndarray,
    event_coords: np.ndarray,
    hospital_coords: np.ndarray,
    injuries_obs: np.ndarray,
    deaths_obs: np.ndarray,
    delay_probs: Optional[np.ndarray] = None,
) -> None:
    n_days = len(events_by_day)
    n_hospitals = hospital_coords.shape[0]

    if delay_probs is None:
        delay_probs = jnp.array([0.5, 0.3, 0.15, 0.05])
        delay_probs = delay_probs / jnp.sum(delay_probs)

    mu_w0 = numpyro.sample("mu_w0", dist.Exponential(0.3))
    sigma_mu_w = numpyro.sample("sigma_mu_w", dist.Exponential(10.0))
    eps_mu_w = numpyro.sample(
        "eps_mu_w", dist.Normal(0.0, sigma_mu_w).expand([n_days - 1]).to_event(1)
    )
    log_mu_w0 = jnp.log(mu_w0 + 1e-8)
    log_mu_w = log_mu_w0 + jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(eps_mu_w)])
    mu_w = numpyro.deterministic("mu_w", jnp.exp(log_mu_w))

    mu_i0 = numpyro.sample("mu_i0", dist.Exponential(0.3))
    sigma_mu_i = numpyro.sample("sigma_mu_i", dist.Exponential(10.0))
    eps_mu_i = numpyro.sample(
        "eps_mu_i", dist.Normal(0.0, sigma_mu_i).expand([n_days - 1]).to_event(1)
    )
    log_mu_i0 = jnp.log(mu_i0 + 1e-8)
    log_mu_i = log_mu_i0 + jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(eps_mu_i)])
    mu_i = numpyro.deterministic("mu_i", jnp.exp(log_mu_i))

    p_late = numpyro.sample("p_late", dist.Beta(2, 10))
    ell = numpyro.sample("ell", dist.Exponential(1.0))
    phi_hosp = numpyro.sample("phi_hosp", dist.Exponential(1.0))
    phi_death = numpyro.sample("phi_death", dist.Exponential(1.0))

    norm_weights = spatial_kernel_weights(jnp.array(event_coords), jnp.array(hospital_coords), ell)
    effective_events = jnp.zeros((n_days, n_hospitals))
    effective_events = effective_events.at[event_day_index].add(norm_weights)

    lam_injuries = mu_w[:, None] * effective_events
    lam_injuries = jnp.maximum(lam_injuries, 1e-8)
    obs_inj = numpyro.sample(
        "obs_injuries",
        dist.GammaPoisson(concentration=phi_hosp, rate=phi_hosp / lam_injuries).to_event(2),
        obs=injuries_obs,
    )

    injuries_to_use = obs_inj if injuries_obs is None else jnp.array(injuries_obs)
    injuries_total = jnp.nansum(injuries_to_use, axis=1)

    delay_len = delay_probs.shape[0]
    pad = jnp.pad(injuries_total, (0, delay_len))
    conv_deaths = jnp.zeros(n_days)
    for k in range(delay_len):
        conv_deaths = conv_deaths + pad[k : n_days + k] * delay_probs[k]

    expected_late_deaths = p_late * conv_deaths
    expected_immediate_deaths = mu_i * jnp.array(events_by_day)
    expected_deaths = expected_immediate_deaths + expected_late_deaths
    expected_deaths = jnp.maximum(expected_deaths, 1e-8)

    numpyro.sample(
        "obs_deaths",
        dist.GammaPoisson(concentration=phi_death, rate=phi_death / expected_deaths),
        obs=deaths_obs,
    )


def casualty_model_with_covariates(
    events_by_day: np.ndarray,
    event_day_index: np.ndarray,
    event_coords: np.ndarray,
    hospital_coords: np.ndarray,
    injuries_obs: np.ndarray,
    deaths_obs: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    delay_probs: Optional[np.ndarray] = None,
) -> None:
    """
    Extended casualty model with time-varying covariates.

    This version allows for covariates (e.g., ceasefire indicators, conflict intensity)
    to modulate the casualty rates over time.

    Parameters
    ----------
    covariates : np.ndarray, optional
        Array of shape (T, K) with K covariates for each day
        Examples: ceasefire indicators, media coverage, intervention events

    Additional Parameters
    ---------------------
    beta : np.ndarray
        Regression coefficients for each covariate (sampled in model)

    Notes
    -----
    Covariates enter the model by modulating mu_w and mu_i:
    - mu_w_t = mu_w * exp(sum_k beta_k * X_tk)
    - mu_i_t = mu_i * exp(sum_k beta_k * X_tk)

    This allows modeling intervention effects similar to epidemia's approach.
    """
    n_days = len(events_by_day)
    n_hospitals = hospital_coords.shape[0]

    if delay_probs is None:
        delay_probs = jnp.array([0.5, 0.3, 0.15, 0.05])
        delay_probs = delay_probs / jnp.sum(delay_probs)

    # Base parameters
    mu_w_base = numpyro.sample("mu_w_base", dist.Exponential(0.3))
    mu_i_base = numpyro.sample("mu_i_base", dist.Exponential(0.3))
    p_late = numpyro.sample("p_late", dist.Beta(2, 10))
    ell = numpyro.sample("ell", dist.Exponential(1.0))

    phi_hosp = numpyro.sample("phi_hosp", dist.Exponential(1.0))
    phi_death = numpyro.sample("phi_death", dist.Exponential(1.0))

    # Covariate effects
    if covariates is not None:
        n_covariates = covariates.shape[1]
        # Sample regression coefficients for each covariate
        beta = numpyro.sample("beta", dist.Normal(0, 1).expand([n_covariates]))

        # Compute time-varying multipliers
        log_multiplier = jnp.dot(jnp.array(covariates), beta)  # shape (T,)
        multiplier = jnp.exp(log_multiplier)

        # Time-varying casualty rates
        mu_w = mu_w_base * multiplier
        mu_i = mu_i_base * multiplier
    else:
        mu_w = mu_w_base
        mu_i = mu_i_base

    # Rest of the model follows the same structure as casualty_model
    norm_weights = spatial_kernel_weights(jnp.array(event_coords), jnp.array(hospital_coords), ell)

    effective_events = jnp.zeros((n_days, n_hospitals))
    effective_events = effective_events.at[event_day_index].add(norm_weights)

    # For time-varying mu_w, need to broadcast correctly
    if covariates is not None:
        lam_injuries = mu_w[:, None] * effective_events  # shape (T, H)
    else:
        lam_injuries = mu_w * effective_events

    lam_injuries = jnp.maximum(lam_injuries, 1e-8)

    numpyro.sample(
        "obs_injuries",
        dist.GammaPoisson(concentration=phi_hosp, rate=phi_hosp / lam_injuries).to_event(2),
        obs=jnp.array(injuries_obs),
    )

    injuries_total = jnp.nansum(jnp.array(injuries_obs), axis=1)

    delay_len = delay_probs.shape[0]
    pad = jnp.pad(injuries_total, (0, delay_len))
    conv_deaths = jnp.zeros(n_days)
    for k in range(delay_len):
        conv_deaths = conv_deaths + pad[k : n_days + k] * delay_probs[k]

    expected_late_deaths = p_late * conv_deaths

    events_by_day_arr = jnp.array(events_by_day)
    if covariates is not None:
        expected_immediate_deaths = mu_i * events_by_day_arr
    else:
        expected_immediate_deaths = mu_i * events_by_day_arr

    expected_deaths = expected_immediate_deaths + expected_late_deaths

    expected_deaths = jnp.maximum(expected_deaths, 1e-8)

    numpyro.sample(
        "obs_deaths",
        dist.GammaPoisson(concentration=phi_death, rate=phi_death / expected_deaths),
        obs=jnp.array(deaths_obs),
    )
