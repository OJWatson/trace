from trace.analysis import (
    create_arviz_inference_data,
    forecast,
    plot_fit,
    plot_forecast,
    posterior_predictive,
    run_inference,
)
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.use("Agg")


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


def test_plot_fit_and_plot_forecast_return_figures(tmp_path):
    dates = np.array([np.datetime64("2023-01-01"),
                     np.datetime64("2023-01-02")])
    injuries_obs = np.array([[1, 0], [0, 1]])
    deaths_obs = np.array([0, 1])

    fig1 = plot_fit(dates, injuries_obs, deaths_obs, preds=None)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    forecast_results = {
        "injuries_median": np.array([1.0, 2.0]),
        "injuries_lower": np.array([0.0, 1.0]),
        "injuries_upper": np.array([2.0, 3.0]),
        "deaths_median": np.array([0.0, 1.0]),
        "deaths_lower": np.array([0.0, 0.0]),
        "deaths_upper": np.array([1.0, 2.0]),
    }
    fig2 = plot_forecast(
        forecast_results, start_date=np.datetime64("2023-01-01"))
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_create_arviz_inference_data_calls_arviz(monkeypatch):
    sentinel = object()

    def _fake_from_numpyro(mcmc, coords=None, dims=None):
        assert coords == {"x": [0]}
        assert dims == {"mu": ["x"]}
        return sentinel

    monkeypatch.setattr("trace.analysis.az.from_numpyro", _fake_from_numpyro)
    out = create_arviz_inference_data(
        mcmc=object(), coords={"x": [0]}, dims={"mu": ["x"]})
    assert out is sentinel


def test_run_inference_smoke_with_dummy_mcmc(monkeypatch):
    class _DummyMCMC:
        def __init__(self, kernel, num_warmup, num_samples, num_chains, progress_bar):
            self.kernel = kernel
            self.ran = False

        def run(self, *args, **kwargs):
            self.ran = True

        def print_summary(self):
            return None

        def get_samples(self):
            return {"mu_w": np.array([1.0]), "mu_i": np.array([0.5]), "p_late": np.array([0.1])}

    monkeypatch.setattr("trace.analysis.NUTS", lambda model: "kernel")
    monkeypatch.setattr("trace.analysis.MCMC", _DummyMCMC)

    mcmc, samples = run_inference(
        events_by_day=np.array([0, 1]),
        event_day_index=np.array([1]),
        event_coords=np.array([[0.0, 0.0]]),
        hospital_coords=np.array([[0.0, 0.0]]),
        injuries_obs=np.array([[0], [1]]),
        deaths_obs=np.array([0, 0]),
        delay_probs=np.array([1.0]),
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        rng_seed=0,
        progress_bar=False,
    )

    assert mcmc.ran is True
    assert "mu_w" in samples
