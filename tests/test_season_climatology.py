"""Climatology: yearly cache, thresholds per (month, hour), the
climatological ensemble, and scoring with both."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from season_helpers import LAT, LON, FakeTruth, make_cfg
from weather_pipeline.season import climatology as CL
from weather_pipeline.season import forecast as F
from weather_pipeline.season import scoring as SC


def fake_temporal(years=(1979, 1980), step_h=3, seed=0):
    """Arraylake single/temporal layout: valid_time/latitude/longitude, a
    global-style 0..360 longitude ordering, sub-daily steps that include
    non-synoptic hours (03, 09, ...) which must be ignored."""
    g = np.random.default_rng(seed)
    t = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31T21", freq=f"{step_h}h")
    lon = np.sort(LON)                                   # 0..11, then 353..359.75
    doy = t.dayofyear.values[:, None, None]
    season = 285 + 10 * np.sin(2 * np.pi * (doy - 110) / 365)
    data = season + g.normal(size=(t.size, LAT.size, lon.size))
    data[np.isin(t.hour, [3, 9, 15, 21])] = 999.0        # must never reach a threshold
    return xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), data.astype("f4"))},
                      coords={"valid_time": t, "latitude": LAT, "longitude": lon})


@pytest.fixture
def clim(tmp_path):
    cfg = make_cfg(tmp_path, thresholds={"quantiles": [0.5, 0.9, 1.0], "climatology_years": [1979, 1980]})
    c = CL.Climatology(cfg, dataset=fake_temporal())
    c.build(workers=2)
    return cfg, c


def test_years_cached_with_synoptic_hours_only(clim):
    cfg, c = clim
    for y in (1979, 1980):
        da = xr.open_dataarray(c.dir / "years" / f"{y}.nc")
        hours = set(pd.DatetimeIndex(da["time"].values).hour)
        assert hours == {0, 6, 12, 18} and float(da.max()) < 999
    s = c.series
    assert s.sizes["time"] == (365 + 366) * 4                       # 1980 is a leap year
    # longitudes back in the forecasts' west-to-east order
    np.testing.assert_array_equal(s["lon"].values, LON)


def test_thresholds_are_month_hour_quantiles(clim):
    cfg, c = clim
    th = c.thresholds()
    assert th.dims == ("quantile", "month", "hour", "lat", "lon")
    assert (th.diff("quantile") >= 0).all()                        # monotone in q
    s = c.series
    t = pd.DatetimeIndex(s["time"].values)
    sel = (t.month == 6) & (t.hour == 12)
    np.testing.assert_allclose(th.sel(quantile=1.0, month=6, hour=12).values, s.values[sel].max(0))
    np.testing.assert_allclose(th.sel(quantile=0.5, month=6, hour=12).values,
                               np.median(s.values[sel], axis=0), rtol=1e-6)
    # thresholds_for picks the valid time's month and hour
    f = c.thresholds_for(["2026-06-24T12", "2026-01-05T00"])
    np.testing.assert_allclose(f.isel(time=0).sel(quantile=1.0), th.sel(quantile=1.0, month=6, hour=12))
    np.testing.assert_allclose(f.isel(time=1).sel(quantile=1.0), th.sel(quantile=1.0, month=1, hour=0))


def test_build_resumes(clim, monkeypatch):
    cfg, c = clim
    (c.dir / "thresholds.nc").unlink()
    again = CL.Climatology(cfg, dataset=fake_temporal())
    monkeypatch.setattr(again, "_fetch_year", lambda y: pytest.fail(f"refetched {y}"))
    again.build()                                                   # years cached: only thresholds
    assert again.is_built()


def test_climatological_ensemble(clim):
    cfg, c = clim
    ens = c.ensemble_for(["2026-06-24T12", "2026-02-28T06"])
    assert ens.dims == ("ensemble", "time", "lat", "lon") and ens.sizes["ensemble"] == 2   # one per year
    s = c.series
    np.testing.assert_allclose(ens.isel(ensemble=1, time=0), s.sel(time="1980-06-24T12"))
    leap = c.ensemble_for(["2028-02-29T00"])                       # 1979 has no 29 Feb
    np.testing.assert_allclose(leap.isel(ensemble=0, time=0), s.sel(time="1979-02-28T00"))
    np.testing.assert_allclose(leap.isel(ensemble=1, time=0), s.sel(time="1980-02-28T00"))


def test_ensemble_subsampled_reproducibly(tmp_path):
    cfg = make_cfg(tmp_path, thresholds={"climatology_years": [1979, 1984]})
    c = CL.Climatology(cfg, dataset=fake_temporal(years=(1979, 1984), step_h=6))
    c.build()
    a = c.ensemble_for(["2026-07-01T12"], n_members=3, seed=5)
    b = c.ensemble_for(["2026-07-01T12"], n_members=3, seed=5)
    assert a.sizes["ensemble"] == 3
    xr.testing.assert_identical(a, b)


# ---- scoring with the climatology -------------------------------------------------

@pytest.fixture
def scored(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path, members=(4, 3), common_members=2, truths=["era5", "ifs_fc0"],
                   period={"start": "2026-07-01", "end": "2026-07-02"},
                   thresholds={"quantiles": [0.5, 0.9, 1.0], "climatology_years": [1979, 1980]})
    F.run_forecasts(cfg)
    sources = {"era5": FakeTruth(offset=-2), "ifs_fc0": FakeTruth(offset=3)}
    return cfg, sources


def run(cfg, sources, clim=None):
    return SC.run_scoring(cfg, truth_sources=sources, mask_source=FakeTruth(), climatology=clim)


def test_scoring_without_climatology_then_with(scored):
    cfg, sources = scored
    first = run(cfg, sources)
    assert first["scored"] == 8 and first["baseline_scored"] == 0
    st = SC.load_statistics(cfg, "era5")
    assert "brier" not in st["aifs-ens1.0"] and SC.BASELINE not in st

    c = CL.Climatology(cfg, dataset=fake_temporal())
    c.build()
    second = run(cfg, sources, c)
    # every model case rescored (now with thresholds) + the baseline for 2 inits x 2 truths
    assert second["scored"] == 8 and second["baseline_scored"] == 4
    assert run(cfg, sources, c)["scored"] == 0                      # then up to date

    st = SC.load_statistics(cfg, "ifs_fc0")
    assert set(st) == {"aifs-ens1.0", "aurora1.5-ens-6h", SC.BASELINE}
    m = SC.season_scores(st["aifs-ens1.0"])
    for key in ("brier", "twcrps", "occurrence_ratio", "miss_rate", "cpit_ks"):
        assert key in m
    assert list(m["quantile"].values) == [0.5, 0.9, 1.0]
    base = SC.season_scores(st[SC.BASELINE])
    assert np.isfinite(base["crps"]).all() and int(st[SC.BASELINE].attrs["n_members"]) == 2


def test_truth_above_the_record_is_an_event(scored):
    """Event counts are consistent with the thresholds: between 0 and the
    weight, and never more events above a higher threshold."""
    cfg, sources = scored
    c = CL.Climatology(cfg, dataset=fake_temporal())
    c.build()
    run(cfg, sources, c)
    st = SC.summed(SC.load_statistics(cfg, "ifs_fc0")["aifs-ens1.0"])
    frac = (st["obs_events"] / st["weight"])
    assert ((frac >= 0) & (frac <= 1)).all()
    assert (frac.diff("quantile") <= 1e-9).all()                    # higher threshold, fewer events
