"""Season scores: cross-checked against scoringrules where it implements the
same estimator, and against properties a calibrated ensemble must satisfy."""

import numpy as np
import pytest
import xarray as xr

from weather_pipeline.season import metrics as S

sr = pytest.importorskip("scoringrules")
rng = np.random.default_rng(42)


def calibrated(m, shape, scale=1.0, seed=0, return_mu=False):
    """Truth and members drawn from the same N(mu, scale^2), mu varying per
    case: the textbook perfectly calibrated ensemble. `scale` may be an
    array broadcastable to `shape` (case-dependent predictability)."""
    g = np.random.default_rng(seed)
    mu = g.normal(size=shape) * 3
    ens = mu + scale * g.normal(size=(m,) + shape)
    y = mu + scale * g.normal(size=shape)
    return (ens, y, mu) if return_mu else (ens, y)


# ---- agreement with scoringrules ------------------------------------------

def test_fair_crps_matches_scoringrules():
    ens, y = rng.normal(size=(7, 50)), rng.normal(size=50)
    ref = sr.crps_ensemble(y, ens, m_axis=0, estimator="fair")
    np.testing.assert_allclose(S.crps_fair(ens, y), ref, rtol=1e-10)


def test_fair_twcrps_matches_scoringrules():
    ens, y = rng.normal(size=(9, 50)), rng.normal(size=50)
    t = 0.7
    ref = sr.twcrps_ensemble(y, ens, m_axis=0, estimator="fair",
                             v_func=lambda z: np.maximum(z, t))
    # censored cases are exactly zero up to rounding, hence atol
    np.testing.assert_allclose(S.twcrps_fair(ens, y, t), ref, rtol=1e-10, atol=1e-12)


def test_weighted_fair_energy_score_matches_scoringrules():
    m, n = 6, 30
    ens, y = rng.normal(size=(m, 4, n)), rng.normal(size=(4, n))
    w = rng.random(n)
    w /= w.sum()
    # the weighted norm equals the plain norm of sqrt(w)-scaled fields
    sw = np.sqrt(w)
    ref = sr.es_ensemble(y * sw, np.moveaxis(ens * sw, 0, 1), m_axis=-2, v_axis=-1,
                         estimator="fair")
    np.testing.assert_allclose(S.energy_score_fair(ens, y, w), ref, rtol=1e-10)


def test_pair_sum_matches_brute_force():
    ens = rng.normal(size=(8, 5))
    brute = sum(np.abs(ens[i] - ens[j]) for i in range(8) for j in range(i + 1, 8))
    np.testing.assert_allclose(S._pair_sum(ens), brute)


# ---- fairness: no dependence on ensemble size ------------------------------

@pytest.mark.parametrize("score", ["crps", "brier"])
def test_fair_scores_do_not_depend_on_ensemble_size(score):
    def mean_score(m):
        ens, y = calibrated(m, (400000,), seed=m)
        return (S.crps_fair(ens, y) if score == "crps"
                else S.brier_fair(ens, y, 1.5)).mean()
    small, large = mean_score(4), mean_score(64)
    assert abs(small - large) / large < 0.03


def test_dss_matches_gaussian_formula():
    ens = rng.normal(2.0, 1.5, size=(100000, 1))
    np.testing.assert_allclose(S.dawid_sebastiani(ens, np.array([3.0])),
                               (1 / 1.5) ** 2 + np.log(1.5 ** 2), rtol=1e-2)


def test_scrps_is_locally_scale_invariant():
    ens, y = rng.normal(size=(10, 100)), rng.normal(size=100)
    c = 7.0
    np.testing.assert_allclose(S.scrps(c * ens, c * y), S.scrps(ens, y) + 0.5 * np.log(c))


# ---- calibration diagnostics on a calibrated ensemble ----------------------

def season_case(m=5, L=3, Y=40, X=50, seed=1, with_thresholds=True, scale=1.0):
    ens, y, mu = calibrated(m, (L, Y, X), scale=scale, seed=seed, return_mu=True)
    fc = xr.DataArray(ens, dims=("ensemble", "lead_time", "lat", "lon"),
                      coords={"lead_time": np.arange(1, L + 1) * np.timedelta64(6, "h")})
    obs = xr.DataArray(y, dims=("lead_time", "lat", "lon"))
    lat = np.linspace(40, 53, Y)
    w = xr.DataArray(np.repeat(np.cos(np.deg2rad(lat))[:, None], X, 1), dims=("lat", "lon"))
    w[:5] = 0                                         # a "sea" strip
    thr = None
    if with_thresholds:
        # thresholds fixed by the TRUE centre, independent of members and
        # truth alike. Deriving them from the ensemble mean (an earlier
        # version of this test) correlates threshold and members and breaks
        # calibration -- the reason the paper fixes thresholds a priori.
        q = np.array([0.5, 1.0, 1.5])
        thr = xr.DataArray(mu[None] + q[:, None, None, None],
                           dims=("quantile", "lead_time", "lat", "lon"),
                           coords={"quantile": [0.7, 0.85, 0.93]})
    return fc, obs, w, thr


def summed(n_cases=40, **kw):
    stats = [S.case_statistics(*season_case(seed=s, **kw), seed=s) for s in range(n_cases)]
    total = sum(stats[1:], stats[0])
    total.attrs = stats[0].attrs
    return S.finalise(total)


@pytest.fixture(scope="module")
def calibrated_scores():
    return summed()


def test_fair_spread_skill_is_one(calibrated_scores):
    # without the (M+1)/M correction a 5-member ensemble would read 0.91
    ssr = calibrated_scores["spread_skill"].values
    np.testing.assert_allclose(ssr, 1.0, atol=0.02)


def test_pit_histogram_is_flat(calibrated_scores):
    np.testing.assert_allclose(calibrated_scores["pit_hist"].values, 1.0, atol=0.08)


def test_tail_calibration(calibrated_scores):
    s = calibrated_scores
    np.testing.assert_allclose(s["occurrence_ratio"].values, 1.0, atol=0.06)
    assert (s["cpit_ks"].values < 0.05).all()


def test_miss_rate_is_one_over_m_plus_one_unconditionally():
    ens, y = calibrated(5, (200000,))
    assert abs(np.mean(y > ens.max(0)) - 1 / 6) < 0.005


def test_binned_spread_skill_on_diagonal():
    """Needs case-dependent predictability and enough members: binning by a
    noisy sample spread flattens the diagram (see case_statistics)."""
    g = np.random.default_rng(7)
    scale = np.exp(g.normal(0, 0.5, size=(1, 40, 50)))
    s = summed(n_cases=20, m=40, with_thresholds=False, scale=scale)
    filled = (s["binned_rmse"].notnull()).values
    sp, rm = s["binned_spread"].values[filled], s["binned_rmse"].values[filled]
    keep = (sp > 0.4) & (sp < 2.5)                 # well-populated bins
    np.testing.assert_allclose(rm[keep], sp[keep], rtol=0.12)


def test_underdispersed_ensemble_is_detected():
    ens, y = calibrated(10, (3, 30, 30), seed=3)
    ens = ens.mean(0) + 0.5 * (ens - ens.mean(0))   # half the spread
    fc = xr.DataArray(ens, dims=("ensemble", "lead_time", "lat", "lon"),
                      coords={"lead_time": np.arange(3) * np.timedelta64(6, "h")})
    obs = xr.DataArray(y, dims=("lead_time", "lat", "lon"))
    w = xr.DataArray(np.ones((30, 30)), dims=("lat", "lon"))
    out = S.finalise(S.case_statistics(fc, obs, w, seed=0))
    assert (out["spread_skill"] < 0.7).all()
    hist = out["pit_hist"].values
    assert (hist[:, [0, -1]] > 2).all()             # U-shaped


def test_masked_cells_do_not_count():
    fc, obs, w, thr = season_case(with_thresholds=False)
    fc[:, :, :5] = np.nan                            # NaN only where weight = 0
    out = S.case_statistics(fc, obs, w)
    assert np.isfinite(out["crps"]).all()


def test_nan_inside_domain_is_an_error():
    fc, obs, w, _ = season_case(with_thresholds=False)
    fc[0, 0, 10, 10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        S.case_statistics(fc, obs, w)


def test_statistics_add_across_cases():
    """Summing two cases then finalising equals scoring them as one dataset."""
    a = S.case_statistics(*season_case(seed=1), seed=0)
    b = S.case_statistics(*season_case(seed=2), seed=0)
    both = a + b
    both.attrs = a.attrs
    crps = S.finalise(both)["crps"]
    expected = (a["crps"] + b["crps"]) / (a["weight"] + b["weight"])
    np.testing.assert_allclose(crps, expected)



# ---- bias and maps ----------------------------------------------------------

def test_bias_is_ensemble_mean_minus_truth():
    fc, obs, w, _ = season_case(with_thresholds=False)
    shifted = fc + 1.5                                   # forecast 1.5 K too warm
    calm = S.finalise(S.case_statistics(fc, obs, w))["bias"]
    warm = S.finalise(S.case_statistics(shifted, obs, w))["bias"]
    np.testing.assert_allclose(warm - calm, 1.5, rtol=1e-10)


def test_maps_by_forecast_day_match_the_cell_scores():
    fc, obs, w, _ = season_case(L=8, with_thresholds=False)   # leads 6..48 h
    st = S.case_statistics(fc, obs, w)
    assert list(st["lead_day"].values) == [1, 2]
    land = w.values > 0
    n = st["map_n"].values                                    # per cell: leads observed that day
    assert (n[:, land] == 4).all() and np.isnan(n[:, ~land]).all()
    maps = S.finalise_maps(st)
    ens = fc.values; y = obs.values
    expected = S.crps_fair(ens, y)[:4].mean(axis=0)          # day 1 = first 4 leads
    np.testing.assert_allclose(maps["crps"].sel(lead_day=1).values[land], expected[land], rtol=1e-5)
    assert np.isnan(maps["crps"].values[:, ~land]).all()     # sea stays NaN
    np.testing.assert_allclose(maps["bias"].sel(lead_day=1).values[land],
                               (ens.mean(0) - y)[:4].mean(0)[land], rtol=1e-4, atol=1e-6)


def test_bands_aggregate_cells_with_area_weights():
    fc, obs, w, _ = season_case(with_thresholds=False)
    st = S.case_statistics(fc, obs, w)
    one_band = xr.full_like(w, "all", dtype=object)
    banded = S.by_band(st, one_band, w)
    whole = S.finalise(st)
    # one band covering the domain = the domain score, averaged over the day's leads
    np.testing.assert_allclose(float(banded["crps"].sel(band="all", lead_day=1)),
                               float((st["crps"].sum() / st["weight"].sum())), rtol=1e-5)
    assert int(banded["cells"].sel(band="all")) == int((w > 0).sum())
    assert np.isfinite(whole["bias"]).all()
