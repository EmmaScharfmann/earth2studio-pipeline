"""Scoring stage with fake forecasts and fake truth sources shaped like the
real ones (global 0.25 deg fields, latitude descending, longitude 0..360)."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weather_pipeline import forecast_runner
from weather_pipeline.season import scoring as SC
from weather_pipeline.season import forecast as F
from weather_pipeline.season.schedule import build_jobs

from season_helpers import FakeTruth, make_cfg

@pytest.fixture
def season(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path, members=(4, 3), common_members=3, truths=["era5", "ifs_fc0"],
                   period={"start": "2026-07-01", "end": "2026-07-02"})
    F.run_forecasts(cfg)
    sources = {"era5": FakeTruth(), "ifs_fc0": FakeTruth(offset=0.5)}
    return cfg, sources


def run(cfg, sources, **kw):
    return SC.run_scoring(cfg, truth_sources=sources, mask_source=FakeTruth(), **kw)


def test_every_case_scored_against_every_truth(season):
    cfg, sources = season
    out = run(cfg, sources)
    assert out["scored"] == 2 * 2 * 2 and out["failed"] == []     # truths x models x inits
    for truth in ("era5", "ifs_fc0"):
        stats = SC.load_statistics(cfg, truth)
        assert set(stats) == {"aifs-ens1.0", "aurora1.5-ens-6h"}
        assert stats["aifs-ens1.0"].sizes["init_time"] == 2


def test_truth_fetched_once_and_shared(season):
    cfg, sources = season
    run(cfg, sources)
    asked = sources["era5"].requested
    assert len(asked) == len(set(asked))                  # no valid time fetched twice
    # 2 inits x 4 leads (6..24 h), overlapping valid times shared
    assert len(asked) == len({pd.Timestamp(j.init_time) + pd.Timedelta(hours=h)
                              for j in build_jobs(cfg) for h in (6, 12, 18, 24)})


def test_rerun_skips_and_changed_forecast_is_rescored(season):
    cfg, sources = season
    run(cfg, sources)
    again = run(cfg, sources)
    assert again["scored"] == 0 and again["up_to_date"] == 8
    job = build_jobs(cfg)[0]
    manifest = forecast_runner.REPO_ROOT / (job.out_zarr + ".manifest.json")
    meta = json.loads(manifest.read_text()); meta["written_utc"] = "2099-01-01T00:00:00"
    manifest.write_text(json.dumps(meta))
    assert run(cfg, sources)["scored"] == 2              # that case, both truths
    assert run(cfg, sources, force=True)["scored"] == 8


def test_land_mask_and_truth_differences_show_up(season):
    cfg, sources = season
    run(cfg, sources)
    w = SC.land_weights(cfg)
    assert 0 < int((w > 0).sum()) < w.size                # the sea strip is excluded
    # fake forecasts ~ N(290, 1); truths differ by a constant 0.5 K
    s5 = SC.season_scores(SC.load_statistics(cfg, "era5")["aifs-ens1.0"])
    sf = SC.season_scores(SC.load_statistics(cfg, "ifs_fc0")["aifs-ens1.0"])
    assert (sf["ens_mean_rmse"] > s5["ens_mean_rmse"]).all()
    assert list(s5["crps"].dims) == ["lead_time"] and np.isfinite(s5["crps"]).all()


def test_season_scores_equal_scoring_the_sum(season):
    cfg, sources = season
    run(cfg, sources)
    st = SC.load_statistics(cfg, "era5")["aifs-ens1.0"]
    whole = SC.season_scores(st)["crps"]
    by_hand = st["crps"].sum("init_time") / st["weight"].sum("init_time")
    np.testing.assert_allclose(whole, by_hand)
    one = SC.season_scores(st, init_times=st["init_time"].values[:1])
    assert np.isfinite(one["crps"]).all()


def test_grid_mismatch_fails_that_case_only(season):
    cfg, sources = season
    sources["era5"] = FakeTruth(lat_shift=0.1)            # not on the 0.25 grid
    out = run(cfg, sources, truths=["era5"])
    assert out["scored"] == 0 and len(out["failed"]) == 4
    assert run(cfg, {"ifs_fc0": sources["ifs_fc0"]}, truths=["ifs_fc0"])["scored"] == 4


def test_unknown_truth_is_rejected(season):
    cfg, sources = season
    with pytest.raises(ValueError, match="not among"):
        run(cfg, sources, truths=["stations"])



def test_bias_maps_and_elevation_bands(season):
    cfg, sources = season
    run(cfg, sources)
    st = SC.load_statistics(cfg, "ifs_fc0")["aifs-ens1.0"]
    maps = SC.season_maps(st)
    w = SC.land_weights(cfg)
    land = (w > 0).values
    assert np.isfinite(maps["crps"].values[:, land]).all()
    assert np.isnan(maps["crps"].values[:, ~land]).all()      # sea NaN after summing cases
    h = SC.elevation(cfg, source=FakeTruth())
    assert float(h.min()) >= 0 and float(h.max()) > 1500
    bands = SC.elevation_bands(h)
    banded = SC.metrics.by_band(SC.summed(st), bands, w)
    # land-only bands, in elevation order; the fake sea strip spans 500-1000 m
    assert list(banded["band"].values) == ["200-500 m", "1000-1500 m", ">1500 m"]
    assert np.isfinite(banded["crps"]).all() and (banded["cells"] > 0).all()


def test_truth_disagreement_from_the_caches(season):
    cfg, sources = season
    run(cfg, sources)
    d = SC.truth_disagreement(cfg, "era5", "ifs_fc0")
    # FakeTruth: ifs_fc0 = era5 + 0.5 K everywhere
    np.testing.assert_allclose(d["mean_diff"], -0.5)
    np.testing.assert_allclose(d["rms_diff"], 0.5)
    land = (SC.land_weights(cfg) > 0).values
    assert np.isnan(d["map_rms_diff"].values[~land]).all()
    assert d.attrs["n_times"] == len(d["time"])


def test_files_from_an_older_stats_version_are_rescored(season):
    cfg, sources = season
    run(cfg, sources)
    path = next((SC._root(cfg) / "scores" / "era5").glob("*.nc"))
    ds = xr.open_dataset(path).load(); ds.attrs["stats_version"] = 1
    ds.to_netcdf(path.with_suffix(".tmp")); path.with_suffix(".tmp").rename(path)
    assert run(cfg, sources)["scored"] == 1


def test_valid_time_window_splits_cases(season):
    cfg, sources = season
    run(cfg, sources)
    st = SC.load_statistics(cfg, "era5")["aifs-ens1.0"]       # inits 1 and 2 July, leads 6..24 h
    inside = SC.valid_time_window(st, "2026-07-02T00", "2026-07-02T23")
    outside = SC.valid_time_window(st, "2026-07-02T00", "2026-07-02T23", inside=False)
    # every (init, lead) pair is in exactly one of the two
    both = SC.summed(inside)["weight"].fillna(0) + SC.summed(outside)["weight"].fillna(0)
    np.testing.assert_allclose(both, SC.summed(st)["weight"])
    # valid on 2 July: 1 July + 24 h, and 2 July + 6, 12, 18 h (+24 h is 3 July)
    n_in = (~np.isnan(inside["weight"])).sum("init_time")
    assert list(n_in.values) == [1, 1, 1, 1]
    assert np.isnan(inside["weight"].sel(init_time="2026-07-02", lead_time=np.timedelta64(24, "h"))).all()
    assert np.isfinite(SC.season_scores(inside)["crps"]).all()
