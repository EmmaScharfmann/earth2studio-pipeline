"""
Probabilistic scores for ensemble forecasts, following Wessel et al.
(arXiv:2610.20519, Section 4.2).

Design
------
1. **Fair estimators throughout** (Ferro 2014): every score estimates the
   score of the ensemble's underlying distribution, so ensembles of
   different sizes are comparable.

2. **Additive statistics.** `case_statistics` reduces one forecast case
   (one init time, all leads) to WEIGHTED SUMS over grid cells: score sums,
   counts, histograms. Sums add across cases, so any aggregate (a season, a
   bootstrap resample, a region, a lead time) is "add the cases, then
   finalise" -- never an average of averages (findings.md: averaging
   per-case RMSEs differed by 0.56 K). Ratio metrics (spread-skill,
   occurrence ratio, miss rate, skill scores) are finalised from sums.

3. **numpy core, xarray at the edge.** Core functions take ensembles with
   the member axis FIRST, shape (M, ...), and observations shaped (...).

Scores
------
overall      fair CRPS, scaled CRPS, Dawid-Sebastiani, ens-mean MSE,
             MSE / frequency of large errors (SE > 4 K^2), fair
             spread-skill ratio (overall and binned by spread), randomised
             PIT histogram, fair energy score (per case, spatial field)
extremes     fair Brier score and fair twCRPS per threshold, occurrence
             ratio, randomised conditional-PIT histogram (-> KS statistic),
             fraction of exceedances above every member (record miss rate
             at the top threshold), ens-mean squared error on exceedances
             (not proper; reported for comparability only)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

LARGE_ERROR_K2 = 4.0          # paper: squared error > 4 K^2
PIT_BINS = 20
CPIT_BINS = 200               # fine bins: KS from the histogram is exact to 1/200
SPREAD_EDGES_K = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf])


# ---------------------------------------------------------------------------
# univariate scores, member axis first
# ---------------------------------------------------------------------------

def _pair_sum(ens: np.ndarray) -> np.ndarray:
    """sum_{i<j} |x_i - x_j| in O(M log M) via the sorted-ensemble identity
    sum_{i<j} (x_(j) - x_(i)) = sum_k x_(k) (2k - M - 1), k = 1..M."""
    m = ens.shape[0]
    xs = np.sort(ens, axis=0)
    k = np.arange(1, m + 1, dtype=float).reshape((m,) + (1,) * (ens.ndim - 1))
    return np.sum(xs * (2 * k - m - 1), axis=0)


def mean_abs_error(ens, y):
    """E|X - y| estimated by the member mean (unbiased)."""
    return np.mean(np.abs(ens - y), axis=0)


def mean_pair_distance(ens):
    """Fair estimate of E|X - X'|: average over the M(M-1) ordered pairs."""
    m = ens.shape[0]
    return 2.0 * _pair_sum(ens) / (m * (m - 1))


def crps_fair(ens, y):
    """Fair CRPS: E|X-y| - 1/2 E|X-X'| with the unbiased pair term."""
    return mean_abs_error(ens, y) - 0.5 * mean_pair_distance(ens)


def scrps(ens, y):
    """Scaled CRPS (Bolin & Wallin 2023) with fair estimates of both
    expectations, as in the paper (the full score keeps a small finite-
    ensemble bias)."""
    ed = mean_pair_distance(ens)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mean_abs_error(ens, y) / ed + 0.5 * np.log(ed)


def twcrps_fair(ens, y, threshold):
    """Threshold-weighted CRPS for w(z) = 1{z >= t}: the chaining function
    v(z) = max(z, t) censors forecast and observation at t."""
    return crps_fair(np.maximum(ens, threshold), np.maximum(y, threshold))


def brier_fair(ens, y, threshold):
    """Fair Brier score for the event y > t (Ferro et al. 2008):
    (p - o)^2 - p(1-p)/(M-1), p the member fraction exceeding t."""
    m = ens.shape[0]
    p = np.mean(ens > threshold, axis=0)
    o = (y > threshold).astype(float)
    return (p - o) ** 2 - p * (1 - p) / (m - 1)


def dawid_sebastiani(ens, y):
    """(y - mu)^2 / s^2 + log s^2 with the unbiased member variance.
    No fair version exists for a general ensemble (paper, Sec. 4.2.2)."""
    mu = ens.mean(axis=0)
    var = ens.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (y - mu) ** 2 / var + np.log(var)


def randomized_rank(ens, y, rng):
    """Rank of y among the members, ties broken at random: integer 0..M.
    Uniform on {0..M} for a calibrated ensemble."""
    below = np.sum(ens < y, axis=0)
    ties = np.sum(ens == y, axis=0)
    return below + np.floor(rng.random(np.shape(y)) * (ties + 1)).astype(int)


def randomized_pit(ens, y, rng):
    """Continuous randomised PIT in [0, 1]: (rank + V) / (M + 1)."""
    m = ens.shape[0]
    return (randomized_rank(ens, y, rng) + rng.random(np.shape(y))) / (m + 1)


def randomized_cpit(ens, y, threshold, rng):
    """Conditional PIT for exceedances y > t (Allen et al. 2025b).

    Among the members above t (m_t of them) plus y, a tail-calibrated
    ensemble makes y's rank uniform on {0..m_t}; (rank + V)/(m_t + 1) is
    then uniform on [0, 1].

    Returns (cpit, defined). Cases with y <= t are not exceedances; cases
    with m_t = 0 (no member above t) have no forecast excess distribution
    and are flagged undefined -- they are counted separately rather than
    filled with uniform noise, which would flatter the KS statistic. Their
    frequency is what the occurrence ratio and miss rate measure.
    """
    exceed = ens > threshold
    m_t = exceed.sum(axis=0)
    rank = np.sum(exceed & (ens < y), axis=0)
    ties = np.sum(exceed & (ens == y), axis=0)
    rank = rank + np.floor(rng.random(np.shape(y)) * (ties + 1)).astype(int)
    cpit = (rank + rng.random(np.shape(y))) / (m_t + 1)
    defined = (y > threshold) & (m_t > 0)
    return cpit, defined


# ---------------------------------------------------------------------------
# multivariate
# ---------------------------------------------------------------------------

def energy_score_fair(ens, y, weights):
    """Fair energy score over a spatial field.

    ens (M, ..., N), y (..., N), weights (N,) normalised to sum to one, so
    ||v||_w = sqrt(sum_i w_i v_i^2) is the area-weighted norm of the paper
    (unequal cells would otherwise skew a plain l2 norm).
    ES = E||X - y|| - 1/2 E||X - X'||, pair term over M(M-1) ordered pairs.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    m = ens.shape[0]
    to_obs = np.sqrt(np.sum(w * (ens - y) ** 2, axis=-1)).mean(axis=0)
    pairs = 0.0
    for i in range(m - 1):
        d = ens[i + 1:] - ens[i]                    # (M-i-1, ..., N)
        pairs = pairs + np.sqrt(np.sum(w * d ** 2, axis=-1)).sum(axis=0)
    return to_obs - pairs / (m * (m - 1))           # 1/2 * 2*sum_{i<j} / (M(M-1))


# ---------------------------------------------------------------------------
# per-case additive statistics
# ---------------------------------------------------------------------------

def _wsum(values, w):
    """Weighted sum over the trailing (lat, lon) axes."""
    return np.sum(values * w, axis=(-2, -1))


def case_statistics(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    weights: xr.DataArray,
    thresholds: xr.DataArray | None = None,
    member_dim: str = "ensemble",
    seed: int = 0,
) -> xr.Dataset:
    """Additive statistics for one forecast case.

    forecast    (member, lead_time, lat, lon), one init time
    obs         (lead_time, lat, lon), truth at the same valid times
    weights     (lat, lon): area weight x land mask; zero excludes a cell
    thresholds  optional (quantile, lead_time, lat, lon): the climatological
                threshold valid for each lead's month and hour, so a single
                `quantile` coordinate covers 0.9 ... 1.0 (record)

    Returns a Dataset with dims lead_time [, quantile] [, bin dims] whose
    variables are weighted SUMS over space. Finalise with `finalise`.
    """
    spatial = [d for d in obs.dims if d != "lead_time"]
    forecast = forecast.transpose(member_dim, "lead_time", *spatial)
    obs = obs.transpose("lead_time", *spatial)
    w2d = weights.transpose(*spatial).values.astype(float)

    ens = forecast.values.astype(float)            # (M, L, Y, X)
    y = obs.values.astype(float)                   # (L, Y, X)
    m = ens.shape[0]
    if m < 2:
        raise ValueError("Ensemble statistics need at least two members")
    use = w2d > 0
    if np.isnan(ens[:, :, use]).any() or np.isnan(y[:, use]).any():
        raise ValueError("NaN inside the weighted domain; mask it via weights")
    ens = np.where(use, ens, 0.0)                  # keep NaNs out of the sums
    y = np.where(use, y, 0.0)
    w = np.broadcast_to(w2d, y.shape)
    rng = np.random.default_rng(seed)

    mu = ens.mean(axis=0)
    var = ens.var(axis=0, ddof=1)
    var = np.where(use, var, 1.0)                  # masked cells: avoid log(0)
    se = (mu - y) ** 2
    large = se > LARGE_ERROR_K2
    lead = forecast["lead_time"].values
    L = len(lead)

    data = {
        "weight": ("lead_time", _wsum(np.ones_like(y), w)),
        "crps": ("lead_time", _wsum(crps_fair(ens, y), w)),
        "scrps": ("lead_time", _wsum(np.where(use, scrps(ens, y), 0.0), w)),
        "dss": ("lead_time", _wsum(np.where(use, dawid_sebastiani(ens, y), 0.0), w)),
        "se": ("lead_time", _wsum(se, w)),
        "var": ("lead_time", _wsum(var, w)),
        "large_se": ("lead_time", _wsum(se * large, w)),
        "large_n": ("lead_time", _wsum(large.astype(float), w)),
        # energy score is one number per (case, lead); stored as a sum over
        # cases with a count so it aggregates like everything else
        "es": ("lead_time", energy_score_fair(
            ens[:, :, use], y[:, use], w2d[use])),
        "es_n": ("lead_time", np.ones(L)),
    }

    # PIT histogram (randomised), weighted
    pit = randomized_pit(ens, y, rng)
    pit_bin = np.minimum((pit * PIT_BINS).astype(int), PIT_BINS - 1)
    hist = np.zeros((L, PIT_BINS))
    for b in range(PIT_BINS):
        hist[:, b] = _wsum((pit_bin == b).astype(float), w)
    data["pit_hist"] = (("lead_time", "pit_bin"), hist)

    # spread-skill binned by ensemble standard deviation. Caveat for small
    # ensembles: the bins use the SAMPLE spread, which is noisy (for
    # Gaussian data it is even independent of the error), so a calibrated
    # 10-member ensemble reads below the diagonal at low spread and above it
    # at high spread. Compare models at the same member count, as the season
    # config enforces with common_members.
    sd_bin = np.digitize(np.sqrt(var), SPREAD_EDGES_K[1:-1])
    nb = len(SPREAD_EDGES_K) - 1
    for name, values in (("bin_weight", np.ones_like(y)), ("bin_var", var), ("bin_se", se)):
        out = np.zeros((L, nb))
        for b in range(nb):
            out[:, b] = _wsum(values * (sd_bin == b), w)
        data[name] = (("lead_time", "spread_bin"), out)

    coords = {
        "lead_time": lead,
        "pit_bin": np.arange(PIT_BINS),
        "spread_bin": np.arange(nb),
        "spread_lower_k": ("spread_bin", SPREAD_EDGES_K[:-1]),
    }

    if thresholds is not None:
        thr_all = thresholds.transpose("quantile", "lead_time", *spatial).values
        Q = thr_all.shape[0]
        tail = {k: np.zeros((L, Q)) for k in (
            "brier", "twcrps", "obs_events", "fc_prob", "miss",
            "exceed_se", "cpit_undefined")}
        cpit_hist = np.zeros((L, Q, CPIT_BINS))
        above_all = y > ens.max(axis=0)
        for q in range(Q):
            # masked cells get a finite dummy threshold: their weight is
            # zero, and a finite value keeps 0 * inf = NaN out of the sums
            t = np.where(use, thr_all[q], 0.0)
            o = (y > t).astype(float)
            tail["brier"][:, q] = _wsum(np.where(use, brier_fair(ens, y, t), 0.0), w)
            tail["twcrps"][:, q] = _wsum(np.where(use, twcrps_fair(ens, y, t), 0.0), w)
            tail["obs_events"][:, q] = _wsum(o, w)
            tail["fc_prob"][:, q] = _wsum(np.mean(ens > t, axis=0), w)
            tail["miss"][:, q] = _wsum(o * above_all, w)
            tail["exceed_se"][:, q] = _wsum(se * o, w)
            cpit, defined = randomized_cpit(ens, y, t, rng)
            tail["cpit_undefined"][:, q] = _wsum(o * ~defined, w)
            cbin = np.minimum((cpit * CPIT_BINS).astype(int), CPIT_BINS - 1)
            for b in range(CPIT_BINS):
                cpit_hist[:, q, b] = _wsum(((cbin == b) & defined).astype(float), w)
        for k, v in tail.items():
            data[k] = (("lead_time", "quantile"), v)
        data["cpit_hist"] = (("lead_time", "quantile", "cpit_bin"), cpit_hist)
        coords["quantile"] = thresholds["quantile"].values
        coords["cpit_bin"] = np.arange(CPIT_BINS)

    ds = xr.Dataset(data, coords=coords)
    ds.attrs["n_members"] = m
    return ds


# ---------------------------------------------------------------------------
# finalisation: sums -> metrics
# ---------------------------------------------------------------------------

def finalise(stats: xr.Dataset) -> xr.Dataset:
    """Metric values from summed statistics (one case or many, already summed
    over whatever dimension is being aggregated, e.g. init_time)."""
    m = stats.attrs["n_members"]
    w = stats["weight"]
    fair = (m + 1) / m           # Fortin et al. 2014: E[SE] = (M+1)/M E[var]
    out = {
        "crps": stats["crps"] / w,
        "scrps": stats["scrps"] / w,
        "dss": stats["dss"] / w,
        "ens_mean_rmse": np.sqrt(stats["se"] / w),
        "large_error_mse": stats["large_se"] / w,
        "large_error_frac": stats["large_n"] / w,
        "spread": np.sqrt(stats["var"] / w),
        "spread_skill": np.sqrt(fair * stats["var"] / stats["se"]),
        "energy_score": stats["es"] / stats["es_n"],
        "pit_hist": stats["pit_hist"] / stats["pit_hist"].sum("pit_bin") * stats.sizes["pit_bin"],
        "binned_spread": np.sqrt(stats["bin_var"] / stats["bin_weight"]),
        "binned_rmse": np.sqrt(stats["bin_se"] / stats["bin_weight"] / fair),
    }
    if "brier" in stats:
        events = stats["obs_events"]
        out.update({
            "brier": stats["brier"] / w,
            "twcrps": stats["twcrps"] / w,
            "event_frac": events / w,
            "occurrence_ratio": events / stats["fc_prob"],
            "miss_rate": stats["miss"] / events,
            "exceed_rmse": np.sqrt(stats["exceed_se"] / events),
            "cpit_undefined_frac": stats["cpit_undefined"] / events,
            "cpit_ks": cpit_ks(stats["cpit_hist"]),
        })
    return xr.Dataset(out)


def cpit_ks(hist: xr.DataArray) -> xr.DataArray:
    """Kolmogorov-Smirnov distance of the CPIT histogram from uniform."""
    cdf = hist.cumsum("cpit_bin") / hist.sum("cpit_bin")
    n = hist.sizes["cpit_bin"]
    uniform = xr.DataArray(np.arange(1, n + 1) / n, dims="cpit_bin")
    return np.abs(cdf - uniform).max("cpit_bin")


def skill_score(model: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """1 - S_model / S_reference, from scores that are already ratios of
    sums over the SAME cases (the paper's definition)."""
    return 1 - model / reference
