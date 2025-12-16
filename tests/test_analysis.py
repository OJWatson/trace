import numpy as np

from trace.analysis import forecast, posterior_predictive


def test_posterior_predictive_shapes():
    samples = {
        "mu_w": np.array([2.0, 2.5]),
        "mu_i": np.array([1.0, 1.2]),
        "p_late": np.array([0.1, 0.2]),
        "ell": np.array([5.0, 6.0]),
    }

    events_by_day = np.array([1, 0, 1])
    event_day_index = np.array([0, 2])
    event_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    hospital_coords = np.array([[0.0, 0.0], [2.0, 0.0]])

    preds = posterior_predictive(
        samples=samples,
        events_by_day=events_by_day,
        event_day_index=event_day_index,
        event_coords=event_coords,
        hospital_coords=hospital_coords,
        injuries_obs_shape=(3, 2),
        deaths_obs_shape=3,
        delay_probs=np.array([1.0]),
        rng_seed=0,
    )

    assert "obs_injuries" in preds
    assert "obs_deaths" in preds

    assert preds["obs_injuries"].shape == (2, 3, 2)
    assert preds["obs_deaths"].shape == (2, 3)


def test_forecast_output_shapes_and_keys():
    np.random.seed(0)

    samples = {
        "mu_w": np.array([2.0] * 5),
        "mu_i": np.array([1.0] * 5),
        "p_late": np.array([0.2] * 5),
    }

    future_events_by_day = np.array([0.0, 1.0, 2.0])
    out = forecast(samples, future_events_by_day, delay_probs=np.array([1.0]))

    for k in [
        "injuries_median",
        "injuries_lower",
        "injuries_upper",
        "deaths_median",
        "deaths_lower",
        "deaths_upper",
    ]:
        assert k in out
        assert len(out[k]) == len(future_events_by_day)
        assert np.all(out[k] >= 0)
