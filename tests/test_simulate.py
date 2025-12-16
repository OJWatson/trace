"""Tests for simulation module."""

import numpy as np
import pytest

from trace.simulate import generate_scenario_events, simulate_conflict_data


def test_simulate_conflict_data_basic():
    """Test basic simulation functionality."""
    delay_probs = np.array([0.5, 0.3, 0.15, 0.05])

    data = simulate_conflict_data(
        n_regions=2,
        n_hospitals=3,
        T=50,
        mu_w_true=5.0,
        mu_i_true=2.0,
        p_late_true=0.2,
        delay_probs=delay_probs,
        events_rate=1.5,
        seed=42,
    )

    # Check output structure
    assert "events" in data
    assert "hospital_incidence" in data
    assert "national_deaths" in data
    assert "region_pop" in data
    assert "hospital_coords" in data

    # Check shapes
    assert data["hospital_incidence"].shape == (50, 3)
    assert data["national_deaths"].shape == (50,)
    assert data["region_pop"].shape == (51, 2)  # T+1 rows
    assert data["hospital_coords"].shape == (3, 2)

    # Check that we have some events
    assert len(data["events"]) > 0

    # Check that deaths and injuries are non-negative
    assert np.all(data["hospital_incidence"] >= 0)
    assert np.all(data["national_deaths"] >= 0)


def test_simulate_reproducibility():
    """Test that simulation is reproducible with same seed."""
    delay_probs = np.array([0.5, 0.3, 0.15, 0.05])

    data1 = simulate_conflict_data(
        n_regions=1,
        n_hospitals=2,
        T=20,
        mu_w_true=3.0,
        mu_i_true=1.0,
        p_late_true=0.15,
        delay_probs=delay_probs,
        seed=123,
    )

    data2 = simulate_conflict_data(
        n_regions=1,
        n_hospitals=2,
        T=20,
        mu_w_true=3.0,
        mu_i_true=1.0,
        p_late_true=0.15,
        delay_probs=delay_probs,
        seed=123,
    )

    # Should produce identical results
    np.testing.assert_array_equal(
        data1["hospital_incidence"], data2["hospital_incidence"])
    np.testing.assert_array_equal(
        data1["national_deaths"], data2["national_deaths"])


def test_generate_scenario_events():
    """Test scenario event generation."""
    # Baseline scenario
    events = generate_scenario_events(T=100, baseline_rate=2.0)
    assert len(events) == 100
    assert np.all(events >= 0)

    # With intervention
    events_with_intervention = generate_scenario_events(
        T=100, baseline_rate=2.0, interventions=[(30, 60, 0.0)]
    )
    assert len(events_with_intervention) == 100

    # With 0.0x rate during intervention, all draws must be zero in that window
    assert np.all(events_with_intervention[30:60] == 0)


def test_simulate_zero_events():
    """Test simulation with zero event rate."""
    delay_probs = np.array([0.5, 0.3, 0.15, 0.05])

    data = simulate_conflict_data(
        n_regions=1,
        n_hospitals=2,
        T=20,
        mu_w_true=3.0,
        mu_i_true=1.0,
        p_late_true=0.15,
        delay_probs=delay_probs,
        events_rate=0.0,  # No events
        seed=42,
    )

    # Should have no events, injuries, or deaths
    assert len(data["events"]) == 0
    assert np.all(data["hospital_incidence"] == 0)
    assert np.all(data["national_deaths"] == 0)
