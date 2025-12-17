"""
Analysis and inference module for TRACE.

This module provides functions for Bayesian inference using MCMC,
posterior predictive checks, forecasting, and visualization.
"""

from typing import Callable, Optional

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive

from trace.model import casualty_model

ModelFn = Callable[..., None]


def run_inference(
    events_by_day: np.ndarray,
    event_day_index: np.ndarray,
    event_coords: np.ndarray,
    hospital_coords: np.ndarray,
    injuries_obs: np.ndarray,
    deaths_obs: np.ndarray,
    delay_probs: Optional[np.ndarray] = None,
    model: ModelFn = casualty_model,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 2,
    rng_seed: int = 0,
    progress_bar: bool = True,
) -> tuple[MCMC, dict[str, np.ndarray]]:
    """
    Run MCMC inference to fit the casualty model to data.

    Uses the No-U-Turn Sampler (NUTS) for efficient Bayesian inference.

    Parameters
    ----------
    events_by_day : np.ndarray
        Array of length T with event counts per day
    event_day_index : np.ndarray
        Array of length E with day indices for each event
    event_coords : np.ndarray
        Array of shape (E, 2) with event coordinates
    hospital_coords : np.ndarray
        Array of shape (H, 2) with hospital coordinates
    injuries_obs : np.ndarray
        Array of shape (T, H) with observed hospital injuries
    deaths_obs : np.ndarray
        Array of length T with observed deaths
    delay_probs : np.ndarray, optional
        Delay distribution from injury to death
    num_warmup : int, default=1000
        Number of warmup (burn-in) iterations
    num_samples : int, default=2000
        Number of posterior samples to draw
    num_chains : int, default=2
        Number of independent MCMC chains
    rng_seed : int, default=0
        Random seed for reproducibility
    progress_bar : bool, default=True
        Whether to show progress bar during sampling

    Returns
    -------
    mcmc : MCMC
        NumPyro MCMC object containing sampling results
    samples : dict
        Dictionary of posterior samples for each parameter

    Examples
    --------
    >>> mcmc, samples = run_inference(
    ...     events_by_day=events,
    ...     event_day_index=event_days,
    ...     event_coords=coords,
    ...     hospital_coords=hosp_coords,
    ...     injuries_obs=injuries,
    ...     deaths_obs=deaths
    ... )
    >>> print(f"Posterior mean mu_w: {samples['mu_w'].mean():.2f}")
    """
    # Set up NUTS sampler
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    # Run MCMC
    mcmc.run(
        jax.random.PRNGKey(rng_seed),
        events_by_day=np.array(events_by_day),
        event_day_index=np.array(event_day_index),
        event_coords=np.array(event_coords) if event_coords is not None else np.zeros((0, 2)),
        hospital_coords=np.array(hospital_coords),
        injuries_obs=np.array(injuries_obs),
        deaths_obs=np.array(deaths_obs),
        delay_probs=np.array(delay_probs) if delay_probs is not None else None,
    )

    # Print summary
    mcmc.print_summary()

    # Get samples
    samples = mcmc.get_samples()

    return mcmc, samples


def posterior_predictive(
    samples: dict[str, np.ndarray],
    events_by_day: np.ndarray,
    event_day_index: np.ndarray,
    event_coords: np.ndarray,
    hospital_coords: np.ndarray,
    injuries_obs_shape: tuple[int, int],
    deaths_obs_shape: int,
    delay_probs: Optional[np.ndarray] = None,
    model: ModelFn = casualty_model,
    rng_seed: int = 1,
) -> dict[str, np.ndarray]:
    """
    Generate posterior predictive samples for model checking.

    Parameters
    ----------
    samples : dict
        Posterior samples from MCMC
    events_by_day : np.ndarray
        Event counts per day
    event_day_index : np.ndarray
        Day indices for events
    event_coords : np.ndarray
        Event coordinates
    hospital_coords : np.ndarray
        Hospital coordinates
    injuries_obs_shape : tuple of int
        Shape (T, H) for injuries observations
    deaths_obs_shape : int
        Length T for deaths observations
    delay_probs : np.ndarray, optional
        Delay distribution
    rng_seed : int, default=1
        Random seed

    Returns
    -------
    dict
        Dictionary with 'obs_injuries' and 'obs_deaths' posterior predictive samples
        - obs_injuries: shape (num_samples, T, H)
        - obs_deaths: shape (num_samples, T)
    """
    # Don't pass observations - let model sample them
    # NumPyro's Predictive will automatically sample from the likelihood when obs=None
    predictive = Predictive(model, samples, return_sites=["obs_injuries", "obs_deaths"])

    preds = predictive(
        jax.random.PRNGKey(rng_seed),
        events_by_day=np.array(events_by_day),
        event_day_index=np.array(event_day_index),
        event_coords=np.array(event_coords) if event_coords is not None else np.zeros((0, 2)),
        hospital_coords=np.array(hospital_coords),
        injuries_obs=None,  # None triggers sampling
        deaths_obs=None,  # None triggers sampling
        delay_probs=np.array(delay_probs) if delay_probs is not None else None,
    )

    return preds


def forecast(
    samples: dict[str, np.ndarray],
    future_events_by_day: np.ndarray,
    delay_probs: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """
    Forecast future casualties and deaths given posterior samples.

    Parameters
    ----------
    samples : dict
        Posterior samples from MCMC inference
    future_events_by_day : np.ndarray
        Array of length H (forecast horizon) with expected events per day
    delay_probs : np.ndarray, optional
        Delay distribution from injury to death

    Returns
    -------
    dict
        Dictionary containing:
        - injuries_median: Median forecast for injuries
        - injuries_lower: 2.5th percentile
        - injuries_upper: 97.5th percentile
        - deaths_median: Median forecast for deaths
        - deaths_lower: 2.5th percentile
        - deaths_upper: 97.5th percentile

    Examples
    --------
    >>> # Forecast 30 days with 2 events per day
    >>> future_events = np.full(30, 2.0)
    >>> forecast_results = forecast(samples, future_events)
    >>> print(f"Expected deaths day 1: {forecast_results['deaths_median'][0]:.1f}")
    """
    if delay_probs is None:
        delay_probs = np.array([0.5, 0.3, 0.15, 0.05])
        delay_probs = delay_probs / delay_probs.sum()

    def _get_last_by_sample(x: np.ndarray) -> np.ndarray:
        # Backwards compatible with scalar parameters:
        # - (S,) -> (S,)
        # - (S, T) -> (S,)
        if x.ndim == 1:
            return x
        if x.ndim == 2:
            return x[:, -1]
        raise ValueError(f"Unsupported parameter array shape: {x.shape}")

    mu_w_last = _get_last_by_sample(np.asarray(samples["mu_w"]))
    mu_i_last = _get_last_by_sample(np.asarray(samples["mu_i"]))
    n_samples = mu_w_last.shape[0]  # number of posterior samples
    horizon = len(future_events_by_day)  # forecast horizon

    sim_deaths = np.zeros((n_samples, horizon))
    sim_injuries = np.zeros((n_samples, horizon))

    delay_len = len(delay_probs)

    # Simulate forward for each posterior sample
    for i in range(n_samples):
        # Start at last inferred day and optionally continue the random walk forward.
        log_mu_w_t = float(np.log(mu_w_last[i] + 1e-12))
        log_mu_i_t = float(np.log(mu_i_last[i] + 1e-12))
        sigma_mu_w = (
            float(np.asarray(samples.get("sigma_mu_w", 0.0))[i]) if "sigma_mu_w" in samples else 0.0
        )
        sigma_mu_i = (
            float(np.asarray(samples.get("sigma_mu_i", 0.0))[i]) if "sigma_mu_i" in samples else 0.0
        )

        p_late = samples["p_late"][i]

        # Queue for delayed deaths
        death_queue = [0] * delay_len

        for t in range(horizon):
            if sigma_mu_w > 0:
                log_mu_w_t += np.random.normal(0.0, sigma_mu_w)
            if sigma_mu_i > 0:
                log_mu_i_t += np.random.normal(0.0, sigma_mu_i)

            mu_w_t = float(np.exp(log_mu_w_t))
            mu_i_t = float(np.exp(log_mu_i_t))
            events_t = future_events_by_day[t]

            # Sample injuries and immediate deaths
            injuries_t = np.random.poisson(mu_w_t * events_t)
            immediate_deaths_t = np.random.poisson(mu_i_t * events_t)

            # Add delayed deaths from queue
            delayed_deaths_t = death_queue[0] if len(death_queue) > 0 else 0

            # Schedule new delayed deaths
            if injuries_t > 0:
                late_deaths_t = np.random.binomial(injuries_t, p_late)
                if late_deaths_t > 0:
                    delays_draw = np.random.multinomial(late_deaths_t, delay_probs)
                    for j in range(delay_len):
                        if j < len(death_queue):
                            death_queue[j] += delays_draw[j]

            # Advance queue
            if len(death_queue) > 0:
                death_queue.pop(0)
                death_queue.append(0)

            # Record totals
            total_deaths_t = immediate_deaths_t + delayed_deaths_t
            sim_injuries[i, t] = injuries_t
            sim_deaths[i, t] = total_deaths_t

    # Compute summary statistics
    return {
        "injuries_median": np.median(sim_injuries, axis=0),
        "injuries_lower": np.percentile(sim_injuries, 2.5, axis=0),
        "injuries_upper": np.percentile(sim_injuries, 97.5, axis=0),
        "deaths_median": np.median(sim_deaths, axis=0),
        "deaths_lower": np.percentile(sim_deaths, 2.5, axis=0),
        "deaths_upper": np.percentile(sim_deaths, 97.5, axis=0),
    }


def plot_fit(
    dates: np.ndarray,
    injuries_obs: np.ndarray,
    deaths_obs: np.ndarray,
    preds: Optional[dict[str, np.ndarray]] = None,
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot model fit against observed data.

    Parameters
    ----------
    dates : np.ndarray
        Array of dates for x-axis
    injuries_obs : np.ndarray
        Observed injuries (T, H)
    deaths_obs : np.ndarray
        Observed deaths (T,)
    preds : dict, optional
        Posterior predictive samples from posterior_predictive()
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot deaths
    axes[0].plot(dates, deaths_obs, "o-", label="Observed deaths", color="darkred", alpha=0.7)

    if preds is not None and "obs_deaths" in preds:
        pred_deaths = np.array(preds["obs_deaths"])
        median_pred = np.median(pred_deaths, axis=0)
        low_pred = np.percentile(pred_deaths, 2.5, axis=0)
        high_pred = np.percentile(pred_deaths, 97.5, axis=0)

        axes[0].fill_between(
            dates, low_pred, high_pred, color="red", alpha=0.2, label="95% credible interval"
        )
        axes[0].plot(dates, median_pred, "-", color="red", label="Posterior median", linewidth=2)

    axes[0].set_ylabel("Daily Deaths", fontsize=12)
    axes[0].legend(loc="best")
    axes[0].set_title("National Deaths: Observed vs Model Fit", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Plot total injuries
    total_inj_obs = np.nansum(injuries_obs, axis=1)
    axes[1].plot(dates, total_inj_obs, "o-", label="Observed injured", color="darkblue", alpha=0.7)

    if preds is not None and "obs_injuries" in preds:
        pred_inj = np.array(preds["obs_injuries"])
        pred_total_inj = np.nansum(pred_inj, axis=2)
        median_inj = np.median(pred_total_inj, axis=0)
        low_inj = np.percentile(pred_total_inj, 2.5, axis=0)
        high_inj = np.percentile(pred_total_inj, 97.5, axis=0)

        axes[1].fill_between(
            dates, low_inj, high_inj, color="blue", alpha=0.2, label="95% credible interval"
        )
        axes[1].plot(dates, median_inj, "-", color="blue", label="Posterior median", linewidth=2)

    axes[1].set_ylabel("Daily Injured (Total)", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].legend(loc="best")
    axes[1].set_title("Hospital Injuries: Observed vs Model Fit", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_forecast(
    forecast_results: dict[str, np.ndarray],
    start_date: np.datetime64,
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot forecasted casualties and deaths.

    Parameters
    ----------
    forecast_results : dict
        Output from forecast() function
    start_date : np.datetime64
        Starting date for forecast
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    horizon = len(forecast_results["deaths_median"])
    dates = [start_date + np.timedelta64(t, "D") for t in range(horizon)]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot forecasted deaths
    axes[0].plot(
        dates, forecast_results["deaths_median"], "-", color="red", label="Median", linewidth=2
    )
    axes[0].fill_between(
        dates,
        forecast_results["deaths_lower"],
        forecast_results["deaths_upper"],
        color="red",
        alpha=0.2,
        label="95% forecast interval",
    )
    axes[0].set_ylabel("Daily Deaths", fontsize=12)
    axes[0].set_title("Forecast - Daily Deaths", fontsize=14, fontweight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # Plot forecasted injuries
    axes[1].plot(
        dates, forecast_results["injuries_median"], "-", color="blue", label="Median", linewidth=2
    )
    axes[1].fill_between(
        dates,
        forecast_results["injuries_lower"],
        forecast_results["injuries_upper"],
        color="blue",
        alpha=0.2,
        label="95% forecast interval",
    )
    axes[1].set_ylabel("Daily Injured", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_title("Forecast - Daily Hospital Injuries", fontsize=14, fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_arviz_inference_data(
    mcmc: MCMC, coords: Optional[dict] = None, dims: Optional[dict] = None
) -> az.InferenceData:
    """
    Convert NumPyro MCMC results to ArviZ InferenceData for diagnostics.

    Parameters
    ----------
    mcmc : MCMC
        NumPyro MCMC object
    coords : dict, optional
        Coordinate labels for dimensions
    dims : dict, optional
        Dimension names for variables

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object for diagnostics and visualization

    Examples
    --------
    >>> idata = create_arviz_inference_data(mcmc)
    >>> az.plot_trace(idata)
    >>> az.summary(idata)
    """
    return az.from_numpyro(mcmc, coords=coords, dims=dims)
