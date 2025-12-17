"""
Basic Usage Example for TRACE

This example demonstrates how to:
1. Simulate synthetic conflict data
2. Fit the Bayesian model using MCMC
3. Perform posterior predictive checks
4. Generate forecasts

This can be converted to a Jupyter notebook for interactive exploration.
"""

import sys

sys.path.insert(0, 'src')

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Non-interactive backend

from trace.analysis import forecast, plot_fit, plot_forecast, posterior_predictive, run_inference
from trace.simulate import simulate_conflict_data

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("TRACE: Basic Usage Example")
print("=" * 70)

# ============================================================================
# Step 1: Generate Synthetic Data
# ============================================================================

print("\n[Step 1] Generating synthetic conflict data...")

# Define true parameters for simulation
TRUE_PARAMS = {
    "mu_w_true": 5.0,  # Average wounded per event
    "mu_i_true": 2.0,  # Average immediate deaths per event
    "p_late_true": 0.2,  # Probability injured person dies
    "ell_true": 20.0,  # Spatial length scale
}

# Delay distribution: 50% die after 1 day, 30% after 2 days, etc.
delay_probs = np.array([0.5, 0.3, 0.15, 0.05])

# Simulate 90 days of conflict with 3 regions and 5 hospitals
sim_data = simulate_conflict_data(
    n_regions=3,
    n_hospitals=5,
    T=90,
    mu_w_true=TRUE_PARAMS["mu_w_true"],
    mu_i_true=TRUE_PARAMS["mu_i_true"],
    p_late_true=TRUE_PARAMS["p_late_true"],
    delay_probs=delay_probs,
    ell_true=TRUE_PARAMS["ell_true"],
    events_rate=2.0,  # Average 2 events per day
    seed=42,
)

print(f"  - Simulated {len(sim_data['events'])} conflict events")
print(f"  - Total injuries: {sim_data['hospital_incidence'].sum()}")
print(f"  - Total deaths: {sim_data['national_deaths'].sum()}")

# ============================================================================
# Step 2: Prepare Data for Modeling
# ============================================================================

print("\n[Step 2] Preparing data for modeling...")

# Extract event information
events_by_day = [0] * 90
event_days = []
event_coords = []

for event in sim_data["events"]:
    day = event["date"]
    events_by_day[day] += 1
    event_days.append(day)
    event_coords.append([event["latitude"], event["longitude"]])

event_coords = np.array(event_coords)
hospital_coords = sim_data["hospital_coords"]

print(f"  - Events by day: {len(events_by_day)} days")
print(f"  - Event coordinates: {event_coords.shape}")
print(f"  - Hospital coordinates: {hospital_coords.shape}")

# ============================================================================
# Step 3: Run Bayesian Inference
# ============================================================================

print("\n[Step 3] Running MCMC inference (this may take a few minutes)...")

mcmc, samples = run_inference(
    events_by_day=events_by_day,
    event_day_index=event_days,
    event_coords=event_coords,
    hospital_coords=hospital_coords,
    injuries_obs=sim_data["hospital_incidence"],
    deaths_obs=sim_data["national_deaths"],
    delay_probs=delay_probs,
    num_warmup=500,  # Reduced for example
    num_samples=1000,  # Reduced for example
    num_chains=2,
    rng_seed=42,
)

print("\n[Results] Posterior estimates vs true values:")
print(f"  mu_w:   {samples['mu_w'].mean():.2f} (true: {TRUE_PARAMS['mu_w_true']:.2f})")
print(f"  mu_i:   {samples['mu_i'].mean():.2f} (true: {TRUE_PARAMS['mu_i_true']:.2f})")
print(f"  p_late: {samples['p_late'].mean():.3f} (true: {TRUE_PARAMS['p_late_true']:.3f})")
print(f"  ell:    {samples['ell'].mean():.2f} (true: {TRUE_PARAMS['ell_true']:.2f})")

# ============================================================================
# Step 4: Posterior Predictive Checks
# ============================================================================

print("\n[Step 4] Generating posterior predictive samples...")

preds = posterior_predictive(
    samples=samples,
    events_by_day=events_by_day,
    event_day_index=event_days,
    event_coords=event_coords,
    hospital_coords=hospital_coords,
    injuries_obs_shape=sim_data["hospital_incidence"].shape,
    deaths_obs_shape=len(sim_data["national_deaths"]),
    delay_probs=delay_probs,
)

print(f"  - Generated {preds['obs_deaths'].shape[0]} posterior predictive samples")

# ============================================================================
# Step 5: Visualize Model Fit
# ============================================================================

print("\n[Step 5] Plotting model fit...")

dates = np.arange(90)

fig = plot_fit(
    dates=dates,
    injuries_obs=sim_data["hospital_incidence"],
    deaths_obs=sim_data["national_deaths"],
    preds=preds,
    save_path="examples/model_fit.png",
)

print("  - Saved plot to examples/model_fit.png")

# ============================================================================
# Step 6: Generate Forecasts
# ============================================================================

print("\n[Step 6] Generating 30-day forecast...")

# Scenario: Assume 1.5 events per day for next 30 days
future_events = np.full(30, 1.5)

forecast_results = forecast(
    samples=samples, future_events_by_day=future_events, delay_probs=delay_probs
)

print(f"  - Forecast median deaths (day 1): {forecast_results['deaths_median'][0]:.1f}")
print(f"  - Forecast median injuries (day 1): {forecast_results['injuries_median'][0]:.1f}")

# Plot forecast
fig_forecast = plot_forecast(
    forecast_results=forecast_results,
    start_date=np.datetime64("2024-01-01"),
    save_path="examples/forecast.png",
)

print("  - Saved forecast plot to examples/forecast.png")

# ============================================================================
# Step 7: Scenario Analysis
# ============================================================================

print("\n[Step 7] Scenario analysis: Impact of ceasefire...")

# Scenario 1: Baseline (1.5 events/day)
baseline_events = np.full(30, 1.5)
baseline_forecast = forecast(samples, baseline_events, delay_probs)

# Scenario 2: Ceasefire (0.2 events/day)
ceasefire_events = np.full(30, 0.2)
ceasefire_forecast = forecast(samples, ceasefire_events, delay_probs)

# Compare total expected deaths
baseline_total = baseline_forecast["deaths_median"].sum()
ceasefire_total = ceasefire_forecast["deaths_median"].sum()
lives_saved = baseline_total - ceasefire_total

print(f"  - Baseline scenario: {baseline_total:.0f} expected deaths")
print(f"  - Ceasefire scenario: {ceasefire_total:.0f} expected deaths")
print(f"  - Lives potentially saved: {lives_saved:.0f}")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
