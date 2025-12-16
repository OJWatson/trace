import pandas as pd
import numpy as np
import pytest

from trace.data import (
    create_hospital_coordinates,
    fetch_acled_data,
    load_hospital_data,
    load_national_deaths,
    prepare_acled_events,
    prepare_mortality_data,
)


class _DummyResponse:
    def __init__(self, json_payload=None, text=""):
        self._json_payload = json_payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_payload


def test_fetch_acled_data_builds_expected_params(monkeypatch):
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _DummyResponse(
            json_payload={
                "data": [
                    {
                        "event_date": "2023-01-01",
                        "fatalities": "1",
                        "latitude": "31.5",
                        "longitude": "34.5",
                    }
                ]
            }
        )

    monkeypatch.setattr("trace.data.requests.get", _fake_get)

    df = fetch_acled_data(
        country="Testland",
        start_date="2023-01-01",
        end_date="2023-01-02",
        api_token="token",
        api_email="email@example.com",
        fields=["event_date", "fatalities"],
    )

    assert captured["url"].endswith("/acled/read")
    assert captured["params"]["country"] == "Testland"
    assert captured["params"]["terms"] == "accept"
    assert captured["params"]["key"] == "token"
    assert captured["params"]["email"] == "email@example.com"
    assert captured["params"]["fields"] == "event_date,fatalities"

    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["event_date"])
    assert df["fatalities"].iloc[0] == 1
    assert df["latitude"].iloc[0] == pytest.approx(31.5)
    assert df["longitude"].iloc[0] == pytest.approx(34.5)


def test_fetch_acled_data_empty_returns_empty_df(monkeypatch, capsys):
    def _fake_get(url, params=None, timeout=None):
        return _DummyResponse(json_payload={"data": []})

    monkeypatch.setattr("trace.data.requests.get", _fake_get)

    df = fetch_acled_data(
        country="Testland",
        start_date="2023-01-01",
        end_date="2023-01-02",
    )

    assert df.empty
    out = capsys.readouterr().out
    assert "Warning: No events found" in out


def test_prepare_acled_events_counts_and_coords():
    df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-03"]),
            "latitude": [31.5, 31.6, 31.7],
            "longitude": [34.5, 34.6, 34.7],
        }
    )

    events_by_day, event_days, coords, dates = prepare_acled_events(
        df, start_date="2023-01-01", end_date="2023-01-02"
    )

    assert len(dates) == 2
    assert len(events_by_day) == 2
    assert events_by_day[0] == 2
    assert events_by_day[1] == 0
    assert event_days == [0, 0]
    assert coords.shape == (2, 2)


def test_create_hospital_coordinates_from_mapping():
    hospital_ids = ["H1", "H2"]
    locations = {"H1": (31.5, 34.5), "H2": (31.6, 34.6)}

    coords = create_hospital_coordinates(hospital_ids, locations=locations)
    assert coords.shape == (2, 2)
    assert coords[0, 0] == pytest.approx(31.5)
    assert coords[1, 1] == pytest.approx(34.6)


def test_load_hospital_data_pivots(tmp_path):
    path = tmp_path / "hosp.csv"
    pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "hospital_id": ["H1", "H2", "H1", "H2"],
            "count": [1, 2, 3, 0],
        }
    ).to_csv(path, index=False)

    df = load_hospital_data(str(path))
    assert df.shape == (2, 2)
    assert df.loc[pd.Timestamp("2023-01-01"), "H1"] == 1
    assert df.loc[pd.Timestamp("2023-01-02"), "H1"] == 3


def test_load_national_deaths(tmp_path):
    path = tmp_path / "deaths.csv"
    pd.DataFrame({"date": ["2023-01-02", "2023-01-01"], "deaths": [2, 1]}).to_csv(
        path, index=False
    )

    series = load_national_deaths(str(path))
    assert len(series) == 2
    assert series.iloc[0] == 1
    assert series.iloc[1] == 2


def test_prepare_mortality_data_uses_extrapolated_and_fills_missing():
    mortality_df = pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2023-01-01", "2023-01-03"]),
            "ext_killed": [1, 3],
        }
    )

    series = prepare_mortality_data(
        mortality_df,
        start_date="2023-01-01",
        end_date="2023-01-03",
        use_extrapolated=True,
    )

    assert len(series) == 3
    assert series.iloc[0] == 1
    assert series.iloc[1] == 0
    assert series.iloc[2] == 3
