"""
Simulation module for generating synthetic conflict casualty data.

This module provides tools to simulate hypothetical conflict scenarios with
multiple regions, hospitals, and realistic casualty dynamics. Useful for
model validation and testing.
"""

from typing import Optional

import numpy as np


def simulate_conflict_data(
    n_regions: int,
    n_hospitals: int,
    T: int,
    mu_w_true: float,
    mu_i_true: float,
    p_late_true: float,
    delay_probs: np.ndarray,
    ell_true: float = 20.0,
    p_immediate_true: Optional[float] = None,
    region_populations: Optional[list[float]] = None,
    birth_rate: float = 0.0,
    migration_rate: float = 0.0,
    events_rate: float = 1.0,
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Simulate synthetic conflict casualty data for a hypothetical country.

    This function generates realistic conflict event data with spatial and temporal
    dynamics, including hospital admissions and mortality with appropriate delays.

    Parameters
    ----------
    n_regions : int
        Number of regions in the country
    n_hospitals : int
        Number of hospitals
    T : int
        Number of days to simulate
    mu_w_true : float
        True average number of wounded per event
    mu_i_true : float
        True average number of immediate deaths per event
    p_late_true : float
        True probability an injured person eventually dies
    delay_probs : np.ndarray
        Array of probabilities for delay 1,2,... days until death (must sum to 1)
    ell_true : float, default=20.0
        True spatial length scale for hospital allocation (in coordinate units)
    p_immediate_true : float, optional
        Probability a casualty is an immediate death. If None, computed from mu_i/total
    region_populations : list of float, optional
        Initial population for each region. If None, uses 100,000 per region
    birth_rate : float, default=0.0
        Daily birth rate per person (fraction)
    migration_rate : float, default=0.0
        Daily net migration rate per person (fraction)
    events_rate : float, default=1.0
        Average number of events per day (Poisson rate)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys: events, hospital_incidence, national_deaths, region_pop,
        hospital_coords, region_centers.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        delay_probs = np.array([0.5, 0.3, 0.15, 0.05])
        data = simulate_conflict_data(
            n_regions=3,
            n_hospitals=5,
            T=100,
            mu_w_true=5.0,
            mu_i_true=2.0,
            p_late_true=0.2,
            delay_probs=delay_probs,
            events_rate=2.0,
            seed=42,
        )

        print(f"Simulated {len(data['events'])} events")
        print(f"Total deaths: {data['national_deaths'].sum()}")

    Notes
    -----
    The simulation process:
    1. Places regions and hospitals in a 100x100 coordinate space
    2. Generates events via Poisson process
    3. Allocates casualties to hospitals via exponential decay kernel
    4. Schedules delayed deaths according to delay distribution
    5. Tracks population changes over time
    """
    rng = np.random.RandomState(
        seed) if seed is not None else np.random.RandomState()

    # ========== Set up spatial layout ==========

    # Place region centers on a grid with jitter
    region_centers = []
    if n_regions > 0:
        side = int(np.ceil(np.sqrt(n_regions)))
        for i in range(n_regions):
            row, col = divmod(i, side)
            center_x = (row + 0.5) * (100.0 / side) + rng.uniform(-2, 2)
            center_y = (col + 0.5) * (100.0 / side) + rng.uniform(-2, 2)
            region_centers.append((center_x, center_y))
    else:
        region_centers = [(50.0, 50.0)]

    # Place hospitals randomly within the country
    hospital_coords = []
    for h in range(n_hospitals):
        hx = rng.uniform(0, 100)
        hy = rng.uniform(0, 100)
        hospital_coords.append((hx, hy))
    hospital_coords = np.array(hospital_coords)

    # ========== Initialize populations ==========

    if region_populations is None:
        region_populations = [100000] * n_regions
    region_populations = np.array(region_populations, dtype=float)

    # Compute p_immediate if not provided
    if p_immediate_true is None:
        total_mean_casualties = mu_i_true + mu_w_true
        p_immediate_true = mu_i_true / \
            total_mean_casualties if total_mean_casualties > 0 else 0.0

    # ========== Initialize output arrays ==========

    events = []
    hospital_incidence = np.zeros((T, n_hospitals), dtype=int)
    national_deaths = np.zeros(T, dtype=int)
    region_pop = np.zeros((T + 1, n_regions), dtype=int)
    region_pop[0] = region_populations.astype(int)

    # ========== Simulate day by day ==========

    for day in range(T):
        # Generate number of events today
        n_events_today = rng.poisson(events_rate)

        if n_events_today == 0:
            # Update population even with no events
            if n_regions > 0:
                growth_factor = 1 + birth_rate + migration_rate
                region_populations = region_populations * growth_factor
                region_pop[day + 1] = np.floor(region_populations).astype(int)
            continue

        # Process each event
        for e in range(n_events_today):
            # Choose region (weighted by population)
            if n_regions > 0:
                region_probs = region_populations / region_populations.sum()
                region_index = rng.choice(n_regions, p=region_probs)
            else:
                region_index = 0

            region_center = region_centers[region_index] if n_regions > 0 else region_centers[0]

            # Event location: random around region center
            ex = rng.normal(region_center[0], scale=5.0)
            ey = rng.normal(region_center[1], scale=5.0)

            # Draw total casualties
            total_casualties = rng.poisson(mu_w_true + mu_i_true)

            if total_casualties == 0:
                continue

            # Split into immediate deaths vs injured
            immediate_deaths = rng.binomial(total_casualties, p_immediate_true)
            injured = total_casualties - immediate_deaths

            # Allocate injured to hospitals via spatial kernel
            if injured > 0:
                dists = np.linalg.norm(
                    hospital_coords - np.array([ex, ey]), axis=1)
                weights = np.exp(-dists / ell_true)

                if weights.sum() == 0:
                    # Assign to nearest hospital
                    nearest_idx = np.argmin(dists)
                    weights = np.zeros_like(dists)
                    weights[nearest_idx] = 1.0
                else:
                    weights = weights / weights.sum()

                # Multinomial allocation
                injured_alloc = rng.multinomial(injured, weights)
                hospital_incidence[day, :] += injured_alloc

            # Record immediate deaths
            national_deaths[day] += immediate_deaths

            # Schedule delayed deaths
            if injured > 0:
                late_deaths = rng.binomial(injured, p_late_true)
                if late_deaths > 0:
                    delays = rng.multinomial(late_deaths, delay_probs)
                    for i, dcount in enumerate(delays):
                        if dcount > 0:
                            death_day = day + i + 1
                            if death_day < T:
                                national_deaths[death_day] += dcount

            # Record event
            events.append(
                {
                    "date": day,
                    "region": region_index,
                    "latitude": ex,
                    "longitude": ey,
                    "casualties": total_casualties,
                    "immediate_deaths": immediate_deaths,
                    "injured": injured,
                }
            )

        # Update populations
        if n_regions > 0:
            region_populations = region_populations * \
                (1 + birth_rate + migration_rate)

            # Subtract immediate deaths from regions
            for event in [ev for ev in events if ev["date"] == day]:
                region = event["region"]
                region_populations[region] -= event["immediate_deaths"]

            region_pop[day + 1] = np.floor(region_populations).astype(int)

    return {
        "events": events,
        "hospital_incidence": hospital_incidence,
        "national_deaths": national_deaths,
        "region_pop": region_pop,
        "hospital_coords": hospital_coords,
        "region_centers": region_centers,
    }


def generate_scenario_events(
    T: int,
    baseline_rate: float,
    interventions: Optional[list[tuple[int, int, float]]] = None,
) -> np.ndarray:
    """
    Generate event counts for scenario-based forecasting.

    Parameters
    ----------
    T : int
        Number of days to generate
    baseline_rate : float
        Baseline event rate (events per day)
    interventions : list of tuple, optional
        List of (start_day, end_day, multiplier) tuples representing
        interventions that change the event rate

    Returns
    -------
    np.ndarray
        Array of length T with event counts per day

    Examples
    --------
    >>> # Baseline of 2 events/day, with ceasefire from day 30-60 (0.1x rate)
    >>> events = generate_scenario_events(
    ...     T=100,
    ...     baseline_rate=2.0,
    ...     interventions=[(30, 60, 0.1)]
    ... )
    """
    rates = np.full(T, baseline_rate)

    if interventions:
        for start, end, multiplier in interventions:
            rates[start:end] *= multiplier

    events = np.random.poisson(rates)
    return events
