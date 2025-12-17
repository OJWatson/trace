"""
Real Data Example for TRACE

This example demonstrates how to:
1. Load real ACLED conflict event data
2. Fetch real mortality data from Tech for Palestine
3. Simulate hospital data (since we don't have real hospital data yet)
4. Fit the Bayesian model using MCMC
5. Generate forecasts

This uses actual conflict and mortality data from Gaza.
"""

import sys

sys.path.insert(0, "src")

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from trace.analysis import forecast, plot_fit, plot_forecast, posterior_predictive, run_inference
from trace.data import (
    fetch_palestine_mortality_data,
    load_example_acled_data,
    prepare_acled_events,
    prepare_mortality_data,
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("TRACE: Real Data Example - Gaza Conflict Analysis")
print("=" * 70)

# ============================================================================
# Step 1: Load Real ACLED Data
# ============================================================================

print("\n[Step 1] Loading real ACLED conflict event data...")

# Load example ACLED data (Gaza/Palestine events)
events_df = load_example_acled_data()
print(f"  - Loaded {len(events_df)} total events")
print(f"  - Date range: {events_df['event_date'].min()} to {events_df['event_date'].max()}")

# Focus on a specific time period (Oct 7, 2023 - Dec 31, 2023)
analysis_start = "2023-10-07"
analysis_end = "2023-12-31"

# Prepare events for modeling
events_by_day, event_days, event_coords, dates = prepare_acled_events(
    events_df, analysis_start, analysis_end
)

print(f"  - Analysis period: {analysis_start} to {analysis_end}")
print(f"  - Events in period: {len(event_days)}")
print(f"  - Days with events: {sum(1 for x in events_by_day if x > 0)}")

# ============================================================================
# Step 2: Fetch Real Mortality Data
# ============================================================================

print("\n[Step 2] Fetching real mortality data from Tech for Palestine...")

mortality_df = fetch_palestine_mortality_data(start_date=analysis_start, end_date=analysis_end)
print(f"  - Loaded {len(mortality_df)} days of mortality data")

# Prepare mortality data for modeling
deaths_obs = prepare_mortality_data(
    mortality_df, analysis_start, analysis_end, use_extrapolated=True
)
print(f"  - Total deaths in period: {deaths_obs.sum():.0f}")
print(f"  - Average daily deaths: {deaths_obs.mean():.1f}")

# ============================================================================
# Step 3: Simulate Hospital Data (Placeholder)
# ============================================================================

print("\n[Step 3] Simulating hospital data (placeholder)...")
print("  NOTE: Real hospital data not available - using simulated data")

# For this example, we'll create plausible hospital data
# In reality, you would load this from actual hospital records

# Assume 5 major hospitals in Gaza
n_hospitals = 5
T = len(dates)

# Create hospital coordinates (approximate Gaza locations)
hospital_coords = np.array(
    [
        [31.5, 34.45],  # Gaza City area
        [31.52, 34.46],  # Gaza City area
        [31.35, 34.30],  # Khan Younis area
        [31.28, 34.25],  # Rafah area
        [31.54, 34.52],  # Northern Gaza
    ]
)

# Simulate hospital injuries based on events (NOT deaths to avoid circular dependency)
# Use a realistic casualty rate per event
# In Gaza conflict: rough estimate is 3-10 casualties per event
injuries_per_event = 5  # Average injuries per conflict event
injuries_obs = np.zeros((T, n_hospitals))

# Distribute injuries based on events and spatial proximity
for i in range(T):
    if events_by_day[i] > 0:
        # Expected injuries for this day based on number of events
        expected_injuries = injuries_per_event * events_by_day[i]
        day_injuries = np.random.poisson(expected_injuries)

        if day_injuries > 0:
            # Distribute among hospitals with some randomness
            # Hospitals closer to events get more casualties
            hospital_probs = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
            hospital_alloc = np.random.multinomial(day_injuries, hospital_probs)
            injuries_obs[i, :] = hospital_alloc

print(f"  - Simulated injuries at {n_hospitals} hospitals")
print(f"  - Total simulated injuries: {injuries_obs.sum():.0f}")

# ============================================================================
# Step 4: Run Bayesian Inference (Simplified for Speed)
# ============================================================================

print("\n[Step 4] Running MCMC inference...")
print("  NOTE: Using reduced samples for demonstration (increase for production)")

# Define delay distribution (based on medical literature)
delay_probs = np.array([0.5, 0.3, 0.15, 0.05])  # 1-4 day delays

try:
    mcmc, samples = run_inference(
        events_by_day=events_by_day,
        event_day_index=event_days,
        event_coords=event_coords,
        hospital_coords=hospital_coords,
        injuries_obs=injuries_obs,
        deaths_obs=deaths_obs.values,
        delay_probs=delay_probs,
        num_warmup=200,  # Reduced for demo
        num_samples=400,  # Reduced for demo
        num_chains=1,  # Single chain for speed
        rng_seed=42,
    )

    print("\n[Results] Posterior estimates:")
    print(
        f"  mu_w (wounded per event):   {samples['mu_w'].mean():.2f} ± {samples['mu_w'].std():.2f}"
    )
    print(
        f"  mu_i (immediate deaths):    {samples['mu_i'].mean():.2f} ± {samples['mu_i'].std():.2f}"
    )
    print(
        f"  p_late (hospital fatality): {samples['p_late'].mean():.3f} ± {samples['p_late'].std():.3f}"
    )
    print(f"  ell (spatial scale):        {samples['ell'].mean():.2f} ± {samples['ell'].std():.2f}")

    # ============================================================================
    # Step 5: Posterior Predictive Checks
    # ============================================================================

    print("\n[Step 5] Generating posterior predictive samples...")

    preds = posterior_predictive(
        samples=samples,
        events_by_day=events_by_day,
        event_day_index=event_days,
        event_coords=event_coords,
        hospital_coords=hospital_coords,
        injuries_obs_shape=injuries_obs.shape,
        deaths_obs_shape=len(deaths_obs),
        delay_probs=delay_probs,
    )

    print(f"  - Generated {preds['obs_deaths'].shape[0]} posterior predictive samples")

    # ============================================================================
    # Step 6: Visualize Model Fit
    # ============================================================================

    print("\n[Step 6] Plotting model fit...")

    fig = plot_fit(
        dates=dates,
        injuries_obs=injuries_obs,
        deaths_obs=deaths_obs.values,
        preds=preds,
        save_path="examples/real_data_fit.png",
    )
    plt.close(fig)

    print("  - Saved plot to examples/real_data_fit.png")

    # ============================================================================
    # Step 7: Generate Forecasts
    # ============================================================================

    print("\n[Step 7] Generating 30-day forecast...")

    # Scenario: Assume similar event rate as last 30 days
    recent_event_rate = np.mean(events_by_day[-30:])
    future_events = np.full(30, recent_event_rate)

    forecast_results = forecast(
        samples=samples, future_events_by_day=future_events, delay_probs=delay_probs
    )

    print(f"  - Forecast median deaths (day 1): {forecast_results['deaths_median'][0]:.1f}")
    print(f"  - Forecast median injuries (day 1): {forecast_results['injuries_median'][0]:.1f}")
    print(f"  - 30-day forecast total deaths: {forecast_results['deaths_median'].sum():.0f}")

    # Plot forecast
    fig_forecast = plot_forecast(
        forecast_results=forecast_results,
        start_date=np.datetime64(analysis_end) + np.timedelta64(1, "D"),
        save_path="examples/real_data_forecast.png",
    )
    plt.close(fig_forecast)

    print("  - Saved forecast plot to examples/real_data_forecast.png")

    # ============================================================================
    # Step 8: Scenario Analysis
    # ============================================================================

    print("\n[Step 8] Scenario analysis: Impact of ceasefire...")

    # Scenario 1: Baseline (continue current rate)
    baseline_events = np.full(30, recent_event_rate)
    baseline_forecast = forecast(samples, baseline_events, delay_probs)

    # Scenario 2: Ceasefire (90% reduction in events)
    ceasefire_events = np.full(30, recent_event_rate * 0.1)
    ceasefire_forecast = forecast(samples, ceasefire_events, delay_probs)

    # Compare
    baseline_total = baseline_forecast["deaths_median"].sum()
    ceasefire_total = ceasefire_forecast["deaths_median"].sum()
    lives_saved = baseline_total - ceasefire_total

    print(f"  - Baseline scenario (current rate): {baseline_total:.0f} expected deaths")
    print(f"  - Ceasefire scenario (90% reduction): {ceasefire_total:.0f} expected deaths")
    print(f"  - Lives potentially saved: {lives_saved:.0f}")

    print("\n" + "=" * 70)
    print("Real data example completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - examples/real_data_fit.png")
    print("  - examples/real_data_forecast.png")

except Exception as e:
    print(f"\n[ERROR] Analysis failed: {e}")
    print("This may be due to data quality issues or numerical instabilities.")
    print("Try adjusting the date range or MCMC parameters.")
    import traceback

    traceback.print_exc()
