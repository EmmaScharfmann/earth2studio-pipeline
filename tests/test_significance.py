"""Paired block bootstrap and Diebold-Mariano tests on stored statistics."""

import numpy as np
import pandas as pd
import xarray as xr

from weather_pipeline.season import metrics as M
from weather_pipeline.season import significance as SG


def fake_stats(n_inits=40, n_leads=4, offset=0.0, rho=0.0, seed=0, noise=0.3):
    """Per-case statistics with known per-init CRPS: weight 1, crps = 1 +
    offset + AR(1) noise with autocorrelation rho across init dates."""
    g = np.random.default_rng(seed)
    e = np.zeros((n_inits, n_leads))
    e[0] = g.normal(size=n_leads)
    for t in range(1, n_inits):
        e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * g.normal(size=n_leads)
    crps = 1.0 + offset + noise * e
    inits = pd.date_range("2026-06-01", periods=n_inits, freq="D")
    leads = np.arange(1, n_leads + 1) * np.timedelta64(24, "h")
    dims = ("init_time", "lead_time")
    ones = np.ones((n_inits, n_leads))
    ds = xr.Dataset({"weight": (dims, ones), "crps": (dims, crps), "scrps": (dims, crps),
                     "dss": (dims, crps), "se": (dims, crps ** 2), "var": (dims, crps ** 2),
                     "err": (dims, 0 * ones), "large_se": (dims, 0 * ones), "large_n": (dims, 0 * ones),
                     "es": (dims, crps), "es_n": (dims, ones),
                     "pit_hist": (dims + ("pit_bin",), np.ones((n_inits, n_leads, 4))),
                     "bin_weight": (dims + ("spread_bin",), np.ones((n_inits, n_leads, 2))),
                     "bin_var": (dims + ("spread_bin",), np.ones((n_inits, n_leads, 2))),
                     "bin_se": (dims + ("spread_bin",), np.ones((n_inits, n_leads, 2))),
                     "map_crps": (("init_time", "lead_day", "lat", "lon"), np.ones((n_inits, 1, 2, 2)))},
                    coords={"init_time": inits, "lead_time": leads})
    ds.attrs["n_members"] = 10
    return ds


def test_block_counts_shape_and_blocks():
    c = SG.block_counts(23, n_boot=500, block=5, seed=1)
    assert c.shape == (500, 23) and (c.sum(axis=1) == 23).all()
    # a single replicate with one block covering everything
    one = SG.block_counts(5, n_boot=3, block=5)
    assert (one == 1).all()


def test_interval_contains_estimate_and_shrinks_with_more_inits():
    small = SG.bootstrap({"m": fake_stats(n_inits=20)}, n_boot=800)["m"]["crps"]
    large = SG.bootstrap({"m": fake_stats(n_inits=200)}, n_boot=800)["m"]["crps"]
    full = M.finalise(fake_stats(n_inits=20).sum("init_time").assign_attrs(n_members=10))["crps"]
    iv = SG.interval(small)
    assert ((iv["low"] <= full) & (full <= iv["high"])).all()
    width = lambda r: (SG.interval(r)["high"] - SG.interval(r)["low"]).mean()
    assert float(width(large)) < 0.5 * float(width(small))


def test_paired_difference_of_a_model_with_itself_is_zero():
    s = fake_stats()
    reps = SG.bootstrap({"a": s, "b": s.copy()}, n_boot=200)
    np.testing.assert_allclose(reps["a"]["crps"] - reps["b"]["crps"], 0)


def test_large_and_cpit_variables_are_not_resampled():
    reps = SG.resampled(fake_stats(), SG.block_counts(40, n_boot=10))
    assert "map_crps" not in reps and reps.sizes["replicate"] == 10


def test_common_inits_only():
    a, b = fake_stats(n_inits=30), fake_stats(n_inits=30).isel(init_time=slice(5, None))
    reps = SG.bootstrap({"a": a, "b": b}, n_boot=50)
    assert reps["a"].sizes["replicate"] == 50


def false_alarms(n, rho, trials=300):
    hits = 0
    for k in range(trials):
        a = fake_stats(n_inits=n, n_leads=1, rho=rho, seed=2 * k)
        b = fake_stats(n_inits=n, n_leads=1, rho=rho, seed=2 * k + 1)
        hits += bool(SG.diebold_mariano(a, b)["p_value"].values[0] < 0.05)
    return hits / trials


def test_dm_false_alarms_uncorrelated_near_nominal():
    assert false_alarms(92, 0.0) < 0.09


def test_dm_false_alarms_persistent_regimes_bounded():
    """Documented residual liberalness (see diebold_mariano): ~10 % at 5 %."""
    assert false_alarms(92, 0.5) < 0.14


def test_dm_detects_a_real_difference_with_sign():
    a = fake_stats(n_inits=60, offset=-0.4, rho=0.5, seed=1, noise=0.2)
    b = fake_stats(n_inits=60, offset=0.0, rho=0.5, seed=2, noise=0.2)
    out = SG.diebold_mariano(a, b)
    assert (out["mean_diff"] < 0).all() and out["significant"].all()
    assert int(out["n_inits"]) == 60 and (out["hac_lag"] >= 4).all()


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6, np.nan])
    # ranks: 0.001*5/1=.005, .008*5/2=.02, .039*5/3=.065, .041*5/4=.051, .6
    np.testing.assert_array_equal(SG.benjamini_hochberg(p, 0.05), [True, True, False, False, False, False])
