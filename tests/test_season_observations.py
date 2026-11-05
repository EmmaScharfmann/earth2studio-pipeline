"""Station truth: QC'd observations, nearest-land-cell mapping, scoring with
time-varying weights, and the gridded truths against the stations."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from season_helpers import FakeTruth, make_cfg
from weather_pipeline.season import forecast as F
from weather_pipeline.season import metrics as M
from weather_pipeline.season import observations as OB
from weather_pipeline.season import scoring as SC

# FakeTruth's land mask: land north of the equator, except a sea strip at
# rows 150-159, i.e. latitudes 52.5 down to 50.25 inside the France box.
STATIONS = pd.DataFrame({
    "station_id": ["inland", "coast", "offshore", "gappy"],
    "name": ["Inland", "Just north of the strip", "Mid strip", "Has gaps"],
    "latitude": [46.0, 52.55, 51.4, 45.0],
    "longitude": [2.0, 2.0, 2.0, 4.0],
    "elevation": [150.0, 20.0, 5.0, 800.0],
    "start_date": pd.Timestamp("1990-01-01"),
    "end_date": pd.NaT,
})


class FakeProvider:
    name, license = "fake", "test"

    def __init__(self, offset=1.0):
        self.offset, self.calls = offset, 0

    def list_stations(self, region, start, end, variables):
        return STATIONS.copy()

    def fetch(self, station_ids, start, end, variables):
        self.calls += 1
        hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
        rows = []
        for sid in station_ids:
            vals = np.full(hours.size, 290.0 + self.offset)
            if sid == "gappy":
                vals[(hours.hour == 12) & (hours.day % 7 == 0)] = np.nan   # a few missing noons
            rows.append(pd.DataFrame({"station_id": sid, "variable": "t2m", "time": hours, "value": vals}))
        df = pd.concat(rows, ignore_index=True)
        return df.dropna(subset=["value"])


@pytest.fixture
def season(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path, members=(4, 3), common_members=3,
                   truths=["era5", "stations", "stations_lapse"],
                   period={"start": "2026-07-01", "end": "2026-07-02"})
    F.run_forecasts(cfg)
    SC.land_weights(cfg, FakeTruth())
    SC.elevation(cfg, source=FakeTruth())
    return cfg


def test_observations_synoptic_and_cached(season):
    prov = FakeProvider()
    st = OB.StationTruth(season, provider=prov)
    obs = st.observations()
    assert set(pd.DatetimeIndex(obs["time"].values).hour) == {0, 6, 12, 18}
    first, last = st.window()
    assert pd.Timestamp(obs["time"].values[-1]) == last          # 2 July + 24 h
    assert prov.calls >= 1
    again = OB.StationTruth(season, provider=FakeProvider()).observations()
    assert again.sizes == obs.sizes                               # read from the season's file


def test_nearest_land_cell_and_dropped_station(season):
    m = OB.StationTruth(season, provider=FakeProvider()).mapping()
    assert m.loc["inland", "mapped"] and m.loc["inland", "distance_km"] < 20
    # nearest cell overall is sea (52.5, ~6 km); the land cell 52.75 (~22 km) is used
    assert m.loc["coast", "mapped"] and m.loc["coast", "cell_lat"] == 52.75
    assert not m.loc["offshore", "mapped"] and "within 30 km" in m.loc["offshore", "reason"]
    np.testing.assert_allclose(m["height_diff"], m["elevation"] - m["cell_height"])


def test_at_stations_and_lapse(season):
    st = OB.StationTruth(season, provider=FakeProvider())
    w = SC.land_weights(season)
    field = xr.DataArray(np.add.outer(w["lat"].values, np.zeros(w.sizes["lon"])), dims=("lat", "lon"),
                         coords={"lat": w["lat"], "lon": w["lon"]})
    at = st.at_stations(field)
    m = st.used()
    np.testing.assert_allclose(at.values, m["cell_lat"].values)       # the cell's own value
    lapsed = st.at_stations(field, lapse=True)
    np.testing.assert_allclose(at - lapsed, OB.STD_LAPSE * m["height_diff"].values)


def test_point_statistics_with_missing_observations():
    """case_statistics on (station,) with time-varying weights equals the
    plain computation over the observed (lead, station) pairs only."""
    g = np.random.default_rng(0)
    ens = 290 + g.normal(size=(5, 3, 6))
    y = 290 + g.normal(size=(3, 6))
    y[1, 2] = np.nan; y[2, 0] = np.nan                                 # missing obs
    fc = xr.DataArray(ens, dims=("ensemble", "lead_time", "station"),
                      coords={"lead_time": np.arange(1, 4) * np.timedelta64(6, "h")})
    obs = xr.DataArray(y, dims=("lead_time", "station"), coords={"lead_time": fc["lead_time"]})
    st = M.case_statistics(fc, obs, obs.notnull().astype(float))
    expected = np.nansum(np.where(np.isnan(y), np.nan, M.crps_fair(ens, np.nan_to_num(y))), axis=1)
    np.testing.assert_allclose(st["crps"], expected)
    assert list(st["weight"].values) == [6, 5, 5]
    assert st["map_n"].dims == ("lead_day", "station") and st["map_n"].sel(station=2).values.item() == 2


def test_scoring_against_stations(season):
    prov = FakeProvider(offset=1.0)
    out = SC.run_scoring(season, truth_sources={"era5": FakeTruth(), "stations": prov, "stations_lapse": prov},
                         mask_source=FakeTruth())
    assert out["failed"] == [] and out["scored"] == 2 * 2 * 3        # models x inits x truths
    st = SC.load_statistics(season, "stations")["aifs-ens1.0"]
    assert st.attrs["n_members"] == 3
    stats = SC.summed(st)
    n_used = len(OB.StationTruth(season, provider=prov).used())
    assert float(stats["weight"].max()) <= 2 * n_used                 # 2 inits
    # forecasts ~ N(290, 1), stations at 291 K: a clear cool bias
    assert (SC.season_scores(st)["bias"] < -0.5).all()
    again = SC.run_scoring(season, truth_sources={"era5": FakeTruth(), "stations": prov, "stations_lapse": prov},
                           mask_source=FakeTruth())
    assert again["scored"] == 0                                       # up to date, no thresholds expected
    maps = SC.season_maps(st)
    assert maps["crps"].dims == ("lead_day", "station")


def test_gridded_truths_against_stations(season):
    prov = FakeProvider(offset=1.0)
    SC.run_scoring(season, truths=["era5"], truth_sources={"era5": FakeTruth()}, mask_source=FakeTruth())
    st = OB.StationTruth(season, provider=prov)
    d = OB.truths_at_stations(season, st=st)
    # ERA5 fake is 290 K everywhere, stations 291 K: nearest-land error -1 K
    np.testing.assert_allclose(d["mean_err"].sel(truth="era5", mapping="nearest_land"), -1.0, atol=1e-5)
    assert set(d["mapping"].values) == {"nearest_land", "lapse"}
    assert d["station_rmse"].dims == ("truth", "mapping", "station")


def test_convergence_adds_the_nearest_station(season):
    from weather_pipeline.season import event as E
    prov = FakeProvider(offset=1.0)
    OB.StationTruth(season, provider=prov).mapping()                 # observations + mapping cached
    sites = {"near inland": (46.02, 2.01), "nowhere": (41.0, 9.0)}
    ds = E.convergence(season, "2026-07-02T12", sites, truth_sources={"era5": FakeTruth()})
    assert list(ds["truth"].values) == ["era5", "stations"]
    obs = ds["observed"].sel(truth="stations")
    assert float(obs.sel(site="near inland")) == pytest.approx(291.0)
    assert np.isnan(float(obs.sel(site="nowhere"))) and np.isnan(float(obs.sel(site="land mean")))
