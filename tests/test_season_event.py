"""Event convergence: forecasts of one valid time from every init reaching it."""

import numpy as np
import pandas as pd
import pytest

from season_helpers import FakeTruth, make_cfg
from weather_pipeline.season import event as E
from weather_pipeline.season import forecast as F

SITES = {"Bordeaux": (44.83, -0.69), "Paris": (48.82, 2.34)}


@pytest.fixture
def season(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path, members=(4, 3), common_members=3,
                   truths=["era5", "ifs_fc0"], lead_hours={"every": 6, "max": 48},
                   period={"start": "2026-07-01", "end": "2026-07-03"})
    F.run_forecasts(cfg)
    from weather_pipeline.season import scoring
    scoring.land_weights(cfg, FakeTruth())                       # cache the mask
    return cfg, {"era5": FakeTruth(), "ifs_fc0": FakeTruth(offset=0.5)}


def test_inits_reaching_a_valid_time(season):
    cfg, _ = season
    jobs = E.reaching(cfg, "2026-07-03T12")
    # leads up to 48 h: inits of 2 July (36 h) and 3 July (12 h), both models
    assert sorted({j.init_time[:10] for j in jobs}) == ["2026-07-02", "2026-07-03"]
    assert len(jobs) == 4


def test_convergence_layout_and_values(season):
    cfg, sources = season
    ds = E.convergence(cfg, "2026-07-03T12", SITES, truth_sources=sources)
    assert ds["forecast"].dims == ("model", "lead_time", "ensemble", "site")
    assert list(ds["site"].values) == ["Bordeaux", "Paris", "land mean"]
    hours = ds["lead_time"].values / np.timedelta64(1, "h")
    assert list(hours) == [12, 36]
    assert ds.sizes["ensemble"] == 3
    assert np.isfinite(ds["forecast"]).all()
    np.testing.assert_allclose(ds["observed"].sel(truth="era5"), 290.0)
    np.testing.assert_allclose(ds["observed"].sel(truth="ifs_fc0"), 290.5)
    # fake forecasts are N(290, 1): the land mean of members is close to 290
    assert abs(float(ds["forecast"].sel(site="land mean").mean()) - 290) < 0.2


def test_site_outside_the_region_is_an_error(season):
    cfg, sources = season
    with pytest.raises(ValueError, match="outside"):
        E.convergence(cfg, "2026-07-03T12", {"Madrid": (40.4, -3.7 - 10)}, truth_sources=sources)


def test_no_forecast_reaching_the_time(season):
    cfg, sources = season
    with pytest.raises(LookupError, match="No finished forecast"):
        E.convergence(cfg, pd.Timestamp("2026-08-01T12"), SITES, truth_sources=sources)
