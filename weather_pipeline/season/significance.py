"""
Uncertainty of season scores: paired block bootstrap and Diebold-Mariano
tests (Wessel et al., Section 4.4).

Both work on the stored per-case statistics, which are sums: a bootstrap
replicate that draws init times with replacement is a weighted sum of the
cases, with weights = how often each init was drawn. All replicates at once
are a single matrix product, counts (replicates x inits) @ stats (inits x
...), and metrics.finalise turns the summed replicates into scores, like any
other aggregate.

Paired: every model is resampled with the SAME draws, so a difference
between two models in a replicate compares identical cases; intervals on
differences and skill scores are therefore much narrower than the two
models' own intervals would suggest.

Blocks: consecutive init dates are strongly correlated (the same weather
regime), so blocks of BLOCK_DAYS consecutive inits are drawn (moving block
bootstrap), as in the paper.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats as sps

from . import metrics

BLOCK_DAYS = 5
N_BOOT = 2000
INTERVAL = (5.0, 95.0)          # 90 % intervals, as in the paper

# Variables metrics.finalise needs that are cheap to resample. The per-cell
# maps and the fine CPIT histogram are left out: thousands of replicates of
# them would take gigabytes, and their intervals are not reported.
SKIP = ("map_", "cpit_hist")


def block_counts(n_inits: int, n_boot: int = N_BOOT, block: int = BLOCK_DAYS,
                 seed: int = 0) -> np.ndarray:
    """(n_boot, n_inits) draw counts of a moving block bootstrap: blocks of
    `block` consecutive inits, started at random, concatenated to n_inits."""
    if n_inits < 1:
        raise ValueError("no init times to resample")
    block = max(1, min(block, n_inits))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_inits / block))
    starts = rng.integers(0, n_inits - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n_inits]
    counts = np.zeros((n_boot, n_inits))
    np.add.at(counts, (np.arange(n_boot)[:, None], idx), 1)
    return counts


def resampled(stats: xr.Dataset, counts: np.ndarray) -> xr.Dataset:
    """Summed statistics for every replicate: dims (replicate, ...)."""
    out = {}
    C = xr.DataArray(counts, dims=("replicate", "init_time"))
    for name, var in stats.data_vars.items():
        if name.startswith(SKIP) or "init_time" not in var.dims:
            continue
        # NaN (e.g. outside a valid-time window) counts as "not there"
        out[name] = xr.dot(C, var.fillna(0), dims="init_time")
    ds = xr.Dataset(out)
    ds.attrs = dict(stats.attrs)
    return ds


def common(stats_by_model: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
    """The models restricted to their common init times, same order."""
    inits = sorted(set.intersection(*(set(s["init_time"].values) for s in stats_by_model.values())))
    if not inits:
        raise ValueError("the models share no init time")
    return {m: s.sel(init_time=inits) for m, s in stats_by_model.items()}


def bootstrap(stats_by_model: dict[str, xr.Dataset], n_boot: int = N_BOOT,
              block: int = BLOCK_DAYS, seed: int = 0) -> dict[str, xr.Dataset]:
    """Finalised scores per replicate for every model, paired draws.
    {model: Dataset with a 'replicate' dim}."""
    paired = common(stats_by_model)
    n = next(iter(paired.values())).sizes["init_time"]
    counts = block_counts(n, n_boot, block, seed)
    return {m: metrics.finalise(resampled(s, counts)) for m, s in paired.items()}


def interval(replicates: xr.DataArray, bounds=INTERVAL) -> xr.Dataset:
    """Percentile interval over the replicate dim."""
    lo, hi = np.nanpercentile(replicates, bounds, axis=replicates.get_axis_num("replicate"))
    dims = [d for d in replicates.dims if d != "replicate"]
    coords = {k: v for k, v in replicates.coords.items() if "replicate" not in v.dims}
    return xr.Dataset({"low": (dims, lo), "high": (dims, hi)}, coords=coords)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------

def per_init_score(stats: xr.Dataset, score: str = "crps") -> xr.DataArray:
    """Area-mean score of each case: (init_time, lead_time)."""
    return stats[score] / stats["weight"]


def _hac_variance(d: np.ndarray, lag: int) -> float:
    """Long-run variance of a series with Bartlett (Newey-West) weights."""
    d = d - d.mean()
    n = d.size
    v = float(d @ d) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * float(d[k:] @ d[:-k]) / n
    return v


def andrews_lag(d: np.ndarray) -> int:
    """Data-driven Bartlett lag (Andrews 1991, AR(1) plug-in)."""
    x = d - d.mean()
    denom = float(x @ x)
    if denom == 0 or x.size < 3:
        return 0
    r = min(max(float(x[1:] @ x[:-1]) / denom, -0.97), 0.97)
    alpha = 4 * r ** 2 / ((1 - r) ** 2 * (1 + r) ** 2)
    return int(np.floor(1.1447 * (alpha * x.size) ** (1 / 3)))


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Which of the p-values are significant at false discovery rate alpha."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(p.shape, dtype=bool)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    passed = ranked <= alpha
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        sig = np.zeros(len(pv), dtype=bool)
        sig[order[: cutoff + 1]] = True
        out[ok] = sig
    return out


def diebold_mariano(stats_a: xr.Dataset, stats_b: xr.Dataset, score: str = "crps",
                    alpha: float = 0.05) -> xr.Dataset:
    """Per lead time: is model A's mean score different from B's?

    d_t = S_A(t) - S_B(t), the difference of the area-mean scores of the two
    models on the same init t. Statistic mean(d) / sqrt(LRV / n), LRV with
    Bartlett weights. Two refinements over the paper's version, because a
    season gives only ~90 strongly autocorrelated daily cases:

    - lag = max(forecast day - 1 + 4, Andrews' data-driven lag). The
      paper's rule alone underestimates LRV when daily score differences
      are persistent, whatever the number of cases.
    - the Harvey-Leybourne-Newbold small-sample correction, with Student-t
      p-values (n - 1 degrees of freedom).

    Measured on synthetic AR(1) score differences (tests/test_significance):
    false alarms at nominal 5 % are 5-7 % for weakly correlated cases and
    8-11 % for lag-1 autocorrelation 0.5 with 30-300 cases (the plain test:
    10-18 %). So the test stays somewhat liberal for persistent regimes:
    read borderline p-values together with the bootstrap intervals.

    `significant` is after Benjamini-Hochberg across lead times at alpha.
    Negative mean_diff: A has the lower (better) score.
    """
    paired = common({"a": stats_a, "b": stats_b})
    a, b = per_init_score(paired["a"], score), per_init_score(paired["b"], score)
    d = (a - b).transpose("init_time", "lead_time")
    leads_h = d["lead_time"].values / np.timedelta64(1, "h")
    mean, stat, p, lags = [], [], [], []
    for k, h in enumerate(leads_h):
        x = d.values[:, k]
        x = x[np.isfinite(x)]
        n = x.size
        if n < 3:
            mean.append(np.nan); stat.append(np.nan); p.append(np.nan); lags.append(0); continue
        lag = max(int(np.ceil(h / 24)) - 1 + 4, andrews_lag(x))
        v = _hac_variance(x, lag)
        t = x.mean() / np.sqrt(v / n) if v > 0 else np.nan
        hh = lag + 1
        t *= np.sqrt(max(n + 1 - 2 * hh + hh * (hh - 1) / n, 1.0) / n)   # HLN
        mean.append(x.mean()); stat.append(t); lags.append(lag)
        p.append(2 * sps.t.sf(abs(t), df=n - 1) if np.isfinite(t) else np.nan)
    return xr.Dataset({
        "mean_diff": ("lead_time", np.array(mean)),
        "statistic": ("lead_time", np.array(stat)),
        "p_value": ("lead_time", np.array(p)),
        "significant": ("lead_time", benjamini_hochberg(np.array(p), alpha)),
        "hac_lag": ("lead_time", np.array(lags)),
        "n_inits": d.sizes["init_time"],
    }, coords={"lead_time": d["lead_time"].values})
