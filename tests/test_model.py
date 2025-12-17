import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers

from trace.model import casualty_model, casualty_model_with_covariates, spatial_kernel_weights


def test_spatial_kernel_weights_shape_and_normalization():
    event_coords = jnp.array([[0.0, 0.0]])
    hospital_coords = jnp.array([[0.0, 0.0], [10.0, 0.0]])

    weights = spatial_kernel_weights(event_coords, hospital_coords, ell=1.0)

    assert weights.shape == (1, 2)
    assert np.all(np.asarray(weights) >= 0)
    np.testing.assert_allclose(np.asarray(weights).sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert float(weights[0, 0]) > float(weights[0, 1])


def test_casualty_model_emits_expected_sites():
    events_by_day = np.array([1, 0])
    event_day_index = np.array([0])
    event_coords = np.array([[0.0, 0.0]])
    hospital_coords = np.array([[0.0, 0.0]])
    injuries_obs = np.array([[1], [0]])
    deaths_obs = np.array([0, 0])

    seeded = handlers.seed(casualty_model, rng_seed=0)
    tr = handlers.trace(seeded).get_trace(
        events_by_day=events_by_day,
        event_day_index=event_day_index,
        event_coords=event_coords,
        hospital_coords=hospital_coords,
        injuries_obs=injuries_obs,
        deaths_obs=deaths_obs,
        delay_probs=np.array([1.0]),
    )

    for name in ["mu_w", "mu_i", "p_late", "ell", "obs_injuries", "obs_deaths"]:
        assert name in tr

    assert tr["obs_injuries"]["type"] == "sample"
    assert tr["obs_deaths"]["type"] == "sample"


def test_casualty_model_uses_default_delay_probs_when_none():
    events_by_day = np.array([1, 0])
    event_day_index = np.array([0])
    event_coords = np.array([[0.0, 0.0]])
    hospital_coords = np.array([[0.0, 0.0]])
    injuries_obs = np.array([[1], [0]])
    deaths_obs = np.array([0, 0])

    seeded = handlers.seed(casualty_model, rng_seed=0)
    tr = handlers.trace(seeded).get_trace(
        events_by_day=events_by_day,
        event_day_index=event_day_index,
        event_coords=event_coords,
        hospital_coords=hospital_coords,
        injuries_obs=injuries_obs,
        deaths_obs=deaths_obs,
        delay_probs=None,
    )

    assert "obs_deaths" in tr


def test_casualty_model_with_covariates_emits_beta_when_covariates_present():
    events_by_day = np.array([1, 0])
    event_day_index = np.array([0])
    event_coords = np.array([[0.0, 0.0]])
    hospital_coords = np.array([[0.0, 0.0]])
    injuries_obs = np.array([[1], [0]])
    deaths_obs = np.array([0, 0])
    covariates = np.ones((2, 1))

    seeded = handlers.seed(casualty_model_with_covariates, rng_seed=0)
    tr = handlers.trace(seeded).get_trace(
        events_by_day=events_by_day,
        event_day_index=event_day_index,
        event_coords=event_coords,
        hospital_coords=hospital_coords,
        injuries_obs=injuries_obs,
        deaths_obs=deaths_obs,
        covariates=covariates,
        delay_probs=None,
    )

    for name in ["mu_w_base", "mu_i_base", "p_late", "ell", "beta", "obs_injuries", "obs_deaths"]:
        assert name in tr
