"""MeteoFranceHourly across wetterdienst releases: renamed parameters, a
values() method vs property, and native units in degC or kelvin."""

import pandas as pd
import pytest

from weather_pipeline.stations import MeteoFranceHourly


class _Frame:
    def __init__(self, df): self._df = df
    def to_pandas(self): return self._df


class _Values:
    def __init__(self, df): self._df = df
    def all(self): return type("R", (), {"df": _Frame(self._df)})()


def _request_returning(df, method_style):
    class Req:
        def filter_by_station_id(self, ids): return self
    req = Req()
    if method_style:
        req.values = lambda: _Values(df)            # new: values()
    else:
        req.values = _Values(df)                    # old: .values
    return req


def raw(parameter, value):
    t = pd.date_range("2026-06-23T12", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({"station_id": "33281001", "parameter": parameter, "timestamp": t,
                         "value": [value] * 3})


@pytest.mark.parametrize("parameter", ["temperature_air_mean_2m", "temperature_air_2m"])
@pytest.mark.parametrize("value, expected_k", [(41.9, 315.05), (315.05, 315.05)])   # degC or K
@pytest.mark.parametrize("method_style", [True, False])
def test_fetch_chunk_normalises(monkeypatch, parameter, value, expected_k, method_style):
    p = MeteoFranceHourly()
    monkeypatch.setattr(p, "_request", lambda *a, **k: _request_returning(raw(parameter, value), method_style))
    out = p._fetch_chunk(["33281001"], pd.Timestamp("2026-06-23"), pd.Timestamp("2026-06-24"), ["t2m"])
    assert len(out) == 3 and set(out["variable"]) == {"t2m"}
    assert out["value"].round(2).eq(expected_k).all()
