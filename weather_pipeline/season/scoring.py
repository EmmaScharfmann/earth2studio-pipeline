"""
Scoring stage: every finished forecast against every truth, saved as
additive per-case statistics.

For each (truth, model, init) the scorer loads the forecast in the standard
layout (forecast.open_forecast), takes the truth at the same valid times on
the same grid, and stores `metrics.case_statistics` as one small netCDF:

    <season dir>/scores/<truth>/<model>_<YYYYmmddTHHMM>.nc

Those files are what the report notebook loads: summing them over any set of
init times and finalising gives any aggregate (season, month, bootstrap
resample) without touching forecasts again.

Truth fields are cached per valid time under <season dir>/truth/<truth>/,
cropped to the region, so the ~20 valid times of a case are fetched once and
shared by every model and every overlapping case.

Weights are cos(latitude) x land, land = IFS land-sea fraction > 0.5 (the
paper's mask, from ECMWF open data, so no Arraylake login is needed for it).

A case is rescored when its forecast changed (the manifest's written_utc
is stored with the statistics), when its file predates the current
metrics.STATS_VERSION, or with force=True.
"""

from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import forecast_runner
from ..forecast_runner import init_slug
from ..regions import crop_indices, parse_region
from . import metrics
from .config import SeasonConfig
from .forecast import open_forecast
from .schedule import Job, build_jobs, is_done, season_dir
from .truth import build_truth

logger = logging.getLogger(__name__)

LAND_FRACTION = 0.5
MASK_TIME = "2026-01-01T00"        # the land-sea mask is static; any date works
GRAVITY = 9.80665
ELEVATION_BANDS_M = [-np.inf, 200, 500, 1000, 1500, np.inf]


def _root(cfg: SeasonConfig) -> Path:
    return forecast_runner.REPO_ROOT / season_dir(cfg)


def stats_path(cfg: SeasonConfig, truth: str, job: Job) -> Path:
    return _root(cfg) / "scores" / truth / f"{job.model}_{init_slug(job.init_time)}.nc"


def _manifest_time(job: Job) -> str | None:
    path = forecast_runner.REPO_ROOT / (job.out_zarr + ".manifest.json")
    try:
        return json.loads(path.read_text()).get("written_utc")
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# truth and weights
# ---------------------------------------------------------------------------

def _crop(da: xr.DataArray, region: str) -> xr.DataArray:
    i, j = crop_indices(parse_region(region), da["lat"].values, da["lon"].values)
    return da.isel(lat=i, lon=j)


class TruthCache:
    """Region-cropped truth fields, one small netCDF per valid time."""

    def __init__(self, cfg: SeasonConfig, truth: str, source=None):
        self.cfg, self.truth = cfg, truth
        self.dir = _root(cfg) / "truth" / truth
        self._source = source

    @property
    def source(self):
        if self._source is None:
            self._source = build_truth(self.truth)
        return self._source

    def _path(self, t: pd.Timestamp) -> Path:
        return self.dir / f"{t:%Y%m%d%H}.nc"

    def fields(self, valid_times) -> xr.DataArray:
        """(lead-ordered) valid times -> DataArray (time, lat, lon)."""
        times = [pd.Timestamp(t) for t in valid_times]
        missing = [t for t in times if not self._path(t).exists()]
        if missing:
            logger.info("%s: fetching %d valid time(s)", self.truth, len(missing))
            da = self.source(missing, [self.cfg.variable]).sel(variable=self.cfg.variable)
            da = _crop(da, self.cfg.region).load()
            self.dir.mkdir(parents=True, exist_ok=True)
            for t in missing:
                field = da.sel(time=np.datetime64(t)).drop_vars(
                    [c for c in da.coords if c not in ("lat", "lon")], errors="ignore")
                tmp = self._path(t).with_suffix(".part")
                field.rename(self.cfg.variable).to_netcdf(tmp)
                tmp.rename(self._path(t))
        out = [xr.open_dataarray(self._path(t)).load() for t in times]
        return xr.concat(out, dim="time").assign_coords(time=np.array(times, "datetime64[ns]"))


def land_weights(cfg: SeasonConfig, mask_source=None) -> xr.DataArray:
    """cos(lat) on land points, 0 elsewhere, on the region's grid. Cached."""
    path = _root(cfg) / "truth" / "land_weights.nc"
    if path.exists():
        return xr.open_dataarray(path).load()
    if mask_source is None:
        from ..ifs_opendata import IFSOpenDataFC0Source
        mask_source = IFSOpenDataFC0Source()
    lsm = _crop(mask_source([MASK_TIME], ["lsm"]).isel(time=0).sel(variable="lsm"), cfg.region)
    lsm = lsm.drop_vars([c for c in lsm.coords if c not in ("lat", "lon")], errors="ignore")
    w = (np.cos(np.deg2rad(lsm["lat"])) * (lsm > LAND_FRACTION)).astype(float)
    w = w.transpose("lat", "lon").rename("weight")
    path.parent.mkdir(parents=True, exist_ok=True)
    w.to_netcdf(path)
    return w


def elevation(cfg: SeasonConfig, source=None) -> xr.DataArray:
    """Surface height (m) on the region's grid, from the IFS surface
    geopotential (same source and grid as the land mask). Cached."""
    path = _root(cfg) / "truth" / "elevation.nc"
    if path.exists():
        return xr.open_dataarray(path).load()
    if source is None:
        from ..ifs_opendata import IFSOpenDataFC0Source
        source = IFSOpenDataFC0Source()
    z = _crop(source([MASK_TIME], ["z_surface"]).isel(time=0).sel(variable="z_surface"), cfg.region)
    z = z.drop_vars([c for c in z.coords if c not in ("lat", "lon")], errors="ignore")
    h = (z / GRAVITY).transpose("lat", "lon").rename("elevation")
    h.attrs["units"] = "m"
    path.parent.mkdir(parents=True, exist_ok=True)
    h.to_netcdf(path)
    return h


def elevation_bands(height: xr.DataArray, edges=ELEVATION_BANDS_M) -> xr.DataArray:
    """Label each cell with its band, e.g. '200-500 m'."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"<{hi:.0f} m" if not np.isfinite(lo) else
                      f">{lo:.0f} m" if not np.isfinite(hi) else f"{lo:.0f}-{hi:.0f} m")
    idx = np.digitize(height.values, edges[1:-1])
    out = xr.DataArray(np.array(labels, dtype=object)[idx], coords=height.coords, dims=height.dims)
    out.attrs["order"] = labels          # low to high, for tables and plots
    return out


def truth_disagreement(cfg: SeasonConfig, a: str = "era5", b: str = "ifs_fc0") -> xr.Dataset:
    """Truth a minus truth b at every valid time cached for both, over the
    land domain: area-weighted mean and RMS difference per valid time, and
    per-cell mean / RMS difference maps. Read from the scorer's truth cache,
    so it covers exactly the valid times that were scored."""
    da, db = _root(cfg) / "truth" / a, _root(cfg) / "truth" / b
    common = sorted({p.name for p in da.glob("*.nc")} & {p.name for p in db.glob("*.nc")})
    if not common:
        raise FileNotFoundError(f"No valid time cached for both {a} and {b}; run --stage score")
    w = land_weights(cfg)
    diffs = []
    for name in common:
        fa, fb = xr.open_dataarray(da / name).load(), xr.open_dataarray(db / name).load()
        diffs.append((fa - _on_grid(fb, fa, b)).expand_dims(
            time=[pd.to_datetime(name.split(".")[0], format="%Y%m%d%H").to_datetime64()]))
    d = xr.concat(diffs, dim="time")
    w = _on_grid(w, d, "land weights")
    land = w > 0
    wsum = w.sum()
    return xr.Dataset({
        "mean_diff": (d * w).sum(("lat", "lon")) / wsum,
        "rms_diff": np.sqrt((d ** 2 * w).sum(("lat", "lon")) / wsum),
        "map_mean_diff": d.mean("time").where(land),
        "map_rms_diff": np.sqrt((d ** 2).mean("time")).where(land),
    }, attrs={"a": a, "b": b, "n_times": len(common)})


def _on_grid(field: xr.DataArray, like: xr.DataArray, what: str) -> xr.DataArray:
    """Select `field` at exactly `like`'s lat/lon (both 0.25 deg grids); a
    mismatch beyond 1e-6 deg is an error rather than a silent nearest pick."""
    out = field.sel(lat=like["lat"].values, lon=like["lon"].values,
                    method="nearest", tolerance=1e-6)
    return out.assign_coords(lat=like["lat"].values, lon=like["lon"].values)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

BASELINE = "climatology"     # the climatological ensemble, scored like a model


def _thresholds(clim, valid, fc: xr.DataArray) -> xr.DataArray | None:
    if clim is None:
        return None
    th = clim.thresholds_for(valid).rename(time="lead_time").assign_coords(lead_time=fc["lead_time"].values)
    return _on_grid(th, fc, "thresholds")


def _case(cfg, model, init, fc, truth, weights, clim, written_utc="") -> xr.Dataset:
    valid = [init + pd.Timedelta(lt) for lt in fc["lead_time"].values]
    obs = truth.fields(valid).rename(time="lead_time").assign_coords(lead_time=fc["lead_time"].values)
    obs = _on_grid(obs, fc, truth.truth)
    w = _on_grid(weights, fc, "land weights")
    seed = zlib.crc32(f"{model}|{init.isoformat()}|{truth.truth}".encode())
    stats = metrics.case_statistics(fc, obs, w, thresholds=_thresholds(clim, valid, fc), seed=seed)
    stats = stats.expand_dims(init_time=[np.datetime64(init, "ns")])
    stats.attrs.update(model=model, truth=truth.truth, init_time=init.isoformat(),
                       forecast_written_utc=written_utc, n_members=int(fc.sizes["ensemble"]),
                       has_thresholds=int(clim is not None))
    return stats


def score_case(cfg: SeasonConfig, job: Job, truth: TruthCache,
               weights: xr.DataArray, clim=None) -> xr.Dataset:
    fc = open_forecast(cfg, job)
    return _case(cfg, job.model, pd.Timestamp(job.init_time), fc, truth, weights, clim,
                 _manifest_time(job) or "")


def score_baseline(cfg: SeasonConfig, init_time, truth: TruthCache,
                   weights: xr.DataArray, clim) -> xr.Dataset:
    """The climatological ensemble for the same init and leads as the models."""
    from .climatology import member_seed
    init = pd.Timestamp(init_time)
    valid = [init + lt for lt in cfg.lead_times]
    ens = clim.ensemble_for(valid, cfg.n_members_scored, seed=member_seed(init))
    fc = ens.rename(time="lead_time").assign_coords(lead_time=cfg.lead_times.values)
    return _case(cfg, BASELINE, init, fc, truth, weights, clim)


def _station_case(cfg, model, init, fc, st, truth_name, written_utc="") -> xr.Dataset:
    """One case against station observations: forecast at the stations' land
    cells (lapse-corrected for stations_lapse), weight 1 where observed."""
    from .observations import STATION_TRUTHS
    valid = [init + pd.Timedelta(lt) for lt in fc["lead_time"].values]
    fc_st = st.at_stations(fc, lapse=STATION_TRUTHS[truth_name])
    obs = (st.observed(valid).rename(time="lead_time")
           .assign_coords(lead_time=fc["lead_time"].values).transpose("lead_time", "station"))
    w = obs.notnull().astype(float)
    seed = zlib.crc32(f"{model}|{init.isoformat()}|{truth_name}".encode())
    stats = metrics.case_statistics(fc_st, obs, w, thresholds=None, seed=seed)
    stats = stats.expand_dims(init_time=[np.datetime64(init, "ns")])
    stats.attrs.update(model=model, truth=truth_name, init_time=init.isoformat(),
                       forecast_written_utc=written_utc, n_members=int(fc.sizes["ensemble"]),
                       has_thresholds=0, n_stations=int(fc_st.sizes["station"]))
    return stats


def score_station_case(cfg, job, st, truth_name) -> xr.Dataset:
    return _station_case(cfg, job.model, pd.Timestamp(job.init_time), open_forecast(cfg, job),
                         st, truth_name, _manifest_time(job) or "")


def score_station_baseline(cfg, init_time, st, truth_name, clim) -> xr.Dataset:
    from .climatology import member_seed
    init = pd.Timestamp(init_time)
    valid = [init + lt for lt in cfg.lead_times]
    ens = clim.ensemble_for(valid, cfg.n_members_scored, seed=member_seed(init))
    fc = ens.rename(time="lead_time").assign_coords(lead_time=cfg.lead_times.values)
    return _station_case(cfg, BASELINE, init, fc, st, truth_name)


def _baseline_path(cfg: SeasonConfig, truth: str, init_time) -> Path:
    return _root(cfg) / "scores" / truth / f"{BASELINE}_{init_slug(str(init_time))}.nc"


def _up_to_date(path: Path, written_utc: str | None, has_clim: bool) -> bool:
    """A stored case is current if it was written by this STATS_VERSION, for
    this forecast, and with thresholds whenever a climatology exists."""
    with xr.open_dataset(path) as old:
        a = old.attrs
        return (a.get("stats_version", 1) == metrics.STATS_VERSION
                and (written_utc is None or a.get("forecast_written_utc") == written_utc)
                and bool(a.get("has_thresholds", 0)) >= has_clim)


def _write(stats: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    stats.to_netcdf(tmp)
    tmp.rename(path)


def run_scoring(cfg: SeasonConfig, models: list[str] | None = None,
                truths: list[str] | None = None, force: bool = False,
                truth_sources: dict | None = None, mask_source=None,
                climatology=None) -> dict:
    """Score every finished forecast against every truth. Returns counts.

    With a built climatology (run_season.py --stage climatology), every case
    also gets the extreme scores, and the climatological ensemble is scored
    as the BASELINE pseudo-model on the same init times. Without one, cases
    are scored without thresholds and rescored once it exists.
    """
    truths = truths or list(cfg.truths)
    unknown = set(truths) - set(cfg.truths)
    if unknown:
        raise ValueError(f"{sorted(unknown)} not among the config's truths {list(cfg.truths)}")
    weights = land_weights(cfg, mask_source)
    logger.info("land weights: %d of %d cells", int((weights > 0).sum()), weights.size)

    if climatology is None:
        from .climatology import Climatology
        candidate = Climatology(cfg)
        climatology = candidate if candidate.is_built() else None
    if climatology is None:
        logger.warning("No climatology built: scoring without extreme thresholds or the "
                       "climatological baseline (run --stage climatology to add them)")
    has_clim = climatology is not None

    finished = [j for j in build_jobs(cfg, models) if is_done(j, cfg)]
    inits = sorted({j.init_time for j in finished})
    summary = {"scored": 0, "up_to_date": 0, "failed": [], "baseline_scored": 0}
    from .observations import STATION_TRUTHS, StationTruth
    stations_cache = {}
    for truth_name in truths:
        source = (truth_sources or {}).get(truth_name)
        if truth_name in STATION_TRUTHS:
            # one StationTruth (observations + mapping) shared by both station truths
            st = stations_cache.setdefault("st", StationTruth(cfg, provider=source))
            work = [(job.label, stats_path(cfg, truth_name, job), _manifest_time(job) or "",
                     lambda job=job, st=st, t=truth_name: score_station_case(cfg, job, st, t))
                    for job in finished]
            if has_clim:
                work += [(f"{BASELINE} @ {init}", _baseline_path(cfg, truth_name, init), None,
                          lambda init=init, st=st, t=truth_name: score_station_baseline(cfg, init, st, t, climatology))
                         for init in inits]
            # station thresholds (each station's own climatology) are not built yet
            expect_thresholds = False
        else:
            truth = TruthCache(cfg, truth_name, source)
            work = [(job.label, stats_path(cfg, truth_name, job), _manifest_time(job) or "",
                     lambda job=job, truth=truth: score_case(cfg, job, truth, weights, climatology))
                    for job in finished]
            if has_clim:
                work += [(f"{BASELINE} @ {init}", _baseline_path(cfg, truth_name, init), None,
                          lambda init=init, truth=truth: score_baseline(cfg, init, truth, weights, climatology))
                         for init in inits]
            expect_thresholds = has_clim
        for label, path, written, compute in work:
            if not force and path.exists() and _up_to_date(path, written, expect_thresholds):
                summary["up_to_date"] += 1
                continue
            try:
                stats = compute()
            except Exception as exc:          # one bad case must not stop the season
                logger.error("FAILED %s vs %s: %s", label, truth_name, exc)
                summary["failed"].append(f"{label} vs {truth_name}: {type(exc).__name__}: {exc}")
                continue
            _write(stats, path)
            summary["baseline_scored" if label.startswith(BASELINE) else "scored"] += 1
            logger.info("scored %s vs %s", label, truth_name)
    summary["finished_forecasts"] = len(finished)
    return summary


# ---------------------------------------------------------------------------
# loading for the report
# ---------------------------------------------------------------------------

def load_statistics(cfg: SeasonConfig, truth: str,
                    models: list[str] | None = None) -> dict[str, xr.Dataset]:
    """{model: Dataset with an init_time dim} of every scored case, plus the
    BASELINE ("climatology") when it was scored. Sum over init_time (or any
    subset of it), then metrics.finalise."""
    out = {}
    names = [spec.name for spec in cfg.models] + [BASELINE]
    for name in names:
        if models and name not in models:
            continue
        files = sorted((_root(cfg) / "scores" / truth).glob(f"{name}_*.nc"))
        if not files:
            continue
        parts = [xr.open_dataset(f).load() for f in files]
        ds = xr.concat(parts, dim="init_time").sortby("init_time")
        ds.attrs = {"n_members": parts[0].attrs["n_members"], "model": name, "truth": truth,
                    "has_thresholds": int(all(p.attrs.get("has_thresholds", 0) for p in parts))}
        out[name] = ds
    return out


def season_scores(stats: xr.Dataset, init_times=None) -> xr.Dataset:
    """Finalised metrics over the given init times (default: all)."""
    total = summed(stats, init_times)
    return metrics.finalise(total)


def summed(stats: xr.Dataset, init_times=None) -> xr.Dataset:
    """Statistics summed over init times. min_count=1 keeps cells that are
    NaN in every case (sea, outside the domain) NaN instead of turning them
    into zeros, which would read as perfect scores on the maps."""
    sel = stats if init_times is None else stats.sel(init_time=init_times)
    total = sel.sum("init_time", min_count=1)
    total.attrs = stats.attrs
    return total


def valid_time_window(stats: xr.Dataset, start, end, inside: bool = True) -> xr.Dataset:
    """Keep only the (init, lead) pairs whose valid time is inside [start,
    end] (or outside, with inside=False); the others become NaN and drop out
    of `summed` / `season_scores`.

    A diagnostic, not a replacement for scores over all cases: selecting cases
    because they turned out extreme favours forecasts that over-predict them
    (the forecaster's dilemma). Report it next to the threshold-based scores.
    Per-cell maps are kept only for cases whose whole forecast day is inside.
    """
    valid = stats["init_time"] + stats["lead_time"]
    in_window = (valid >= np.datetime64(pd.Timestamp(start))) & (valid <= np.datetime64(pd.Timestamp(end)))
    keep = in_window if inside else ~in_window
    # a forecast day's map counts only if all its leads are on the kept side
    day = np.ceil(stats["lead_time"] / np.timedelta64(1, "h") / 24).rename("lead_day")
    keep_day = keep.groupby(day).all()
    out = stats.copy()
    for name, var in stats.data_vars.items():
        if "lead_time" in var.dims:
            out[name] = var.where(keep)
        elif "lead_day" in var.dims:
            out[name] = var.where(keep_day)
    return out


def season_maps(stats: xr.Dataset, init_times=None) -> xr.Dataset:
    """Per-cell metrics by forecast day over the given init times."""
    return metrics.finalise_maps(summed(stats, init_times))
