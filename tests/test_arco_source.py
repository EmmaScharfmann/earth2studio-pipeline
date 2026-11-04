"""ARCO-ERA5 source on a small dataset laid out like the real store: one
dataset holding surface and pressure-level fields under CF long names, a
time axis that continues past the end of ERA5T with NaN chunks."""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weather_pipeline.datasource import ARCOERA5Source

LAT = np.array([90.0, 45.0, 0.0, -45.0, -90.0])
LON = np.array([0.0, 90.0, 180.0, 270.0])
LEV = np.array([500, 850])
TIME = pd.date_range("2026-08-30", "2026-09-03T23", freq="h")      # ERA5T ends 2026-09-01


def fake_arco():
    shape2 = (TIME.size, LAT.size, LON.size)
    t2m = np.full(shape2, 290.0)
    t2m[TIME > pd.Timestamp("2026-09-01T23")] = np.nan              # empty tail
    ds = xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), t2m),
            "geopotential_at_surface": (("time", "latitude", "longitude"), np.full(shape2, 111.0)),
            "land_sea_mask": (("time", "latitude", "longitude"), np.ones(shape2)),
            "sea_ice_cover": (("time", "latitude", "longitude"), np.full(shape2, 0.25)),
            "standard_deviation_of_orography": (("time", "latitude", "longitude"), np.full(shape2, 5.0)),
            "slope_of_sub_gridscale_orography": (("time", "latitude", "longitude"), np.full(shape2, 0.1)),
            "geopotential": (("time", "level", "latitude", "longitude"),
                             np.broadcast_to(LEV[None, :, None, None] * 10.0,
                                             (TIME.size, LEV.size, LAT.size, LON.size)).copy()),
        },
        coords={"time": TIME, "latitude": LAT, "longitude": LON, "level": LEV},
        attrs={"valid_time_start": "1940-01-01", "valid_time_stop": "2026-08-31",
               "valid_time_stop_era5t": "2026-09-01"},
    )
    return ds


@pytest.fixture
def src():
    return ARCOERA5Source(fake_arco())


def test_layout_matches_the_pipeline_convention(src):
    da = src(["2026-09-01T12"], ["t2m"])
    assert da.dims == ("time", "variable", "lat", "lon")
    assert da["lat"].values[0] == 90 and float(da.mean()) == 290.0


def test_surface_geopotential_is_not_the_pressure_level_field(src):
    """ARCO's "geopotential" is the 4-D pressure-level field; z_surface must
    resolve to geopotential_at_surface instead."""
    da = src(["2026-09-01T00"], ["z_surface", "z500", "z850"])
    assert da.sel(variable="z_surface").values.max() == 111.0
    assert da.sel(variable="z500").values.max() == 5000.0
    assert da.sel(variable="z850").values.max() == 8500.0


def test_times_after_era5t_are_refused(src):
    src(["2026-09-01T23"], ["t2m"])                       # last covered hour
    with pytest.raises(KeyError, match="ends at 2026-09-01 23:00"):
        src(["2026-09-01T18", "2026-09-02T00"], ["t2m"])


def test_era5t_use_is_logged(src, caplog):
    with caplog.at_level(logging.INFO, logger="weather_pipeline.datasource"):
        src(["2026-08-31T12"], ["t2m"])
        assert "ERA5T" not in caplog.text                  # still final ERA5
        src(["2026-09-01T12"], ["t2m"])
        assert "ERA5T" in caplog.text


def test_store_without_end_date_is_rejected():
    ds = fake_arco()
    ds.attrs = {}
    with pytest.raises(ValueError, match="valid_time_stop"):
        ARCOERA5Source(ds)


def test_sea_ice_comes_from_the_open_store(src, monkeypatch):
    """Must not reopen ARCO through sea_ice.py."""
    import weather_pipeline.sea_ice as sea_ice
    monkeypatch.setattr(sea_ice, "fetch_sea_ice",
                        lambda *a, **k: pytest.fail("reopened ARCO via sea_ice.py"))
    da = src(["2026-09-01T06"], ["sic", "t2m"])
    assert float(da.sel(variable="sic").max()) == 0.25


def test_aifs_statics_are_surface_fields_at_a_covered_time(src):
    """The AIFS constant fields: surface geopotential, not the 37-level
    one, read inside final ERA5 (here the first hour is NaN on purpose)."""
    from weather_pipeline.statics import load_statics
    src.single["land_sea_mask"][0] = np.nan
    out = load_statics(src)
    assert set(out) == {"z", "lsm", "sdor", "slor"}
    for name, da in out.items():
        assert da.dims == ("lat", "lon"), name
        assert not np.isnan(da.values).any(), name
    assert float(out["z"].max()) == 111.0


def test_static_with_extra_dims_is_rejected(src, monkeypatch):
    import weather_pipeline.statics as statics
    monkeypatch.setitem(statics.STATIC_ALIASES, "z", ["geopotential"])   # the old bug
    with pytest.raises(ValueError, match="expected .lat, lon."):
        statics.load_statics(src)
