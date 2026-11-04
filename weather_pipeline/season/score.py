"""
Scoring stage: every finished forecast against every truth, saved as
additive per-case statistics.

For each (truth, model, init) the scorer loads the forecast in the standard
layout (forecast.open_forecast), takes the truth at the same valid times on
the same grid, and stores `scores.case_statistics` as one small netCDF:

    <season dir>/scores/<truth>/<model>_<YYYYmmddTHHMM>.nc

Those files are what the report notebook loads: summing them over any set of
init times and finalising gives any aggregate (season, month, bootstrap
resample) without touching forecasts again.

Truth fields are cached per valid time under <season dir>/truth/<truth>/,
cropped to the region, so the ~20 valid times of a case are fetched once and
shared by every model and every overlapping case.

Weights are cos(latitude) x land, land = IFS land-sea fraction > 0.5 (the
paper's mask, from ECMWF open data, so no Arraylake login is needed for it).

A case is rescored only when its forecast changed (the manifest's
written_utc is stored with the statistics) or with force=True.
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
from . import scores
from .config import SeasonConfig
from .forecast import open_forecast
from .schedule import Job, build_jobs, is_done, season_dir
from .truth import build_truth

logger = logging.getLogger(__name__)

LAND_FRACTION = 0.5
MASK_TIME = "2026-01-01T00"        # the land-sea mask is static; any date works


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


def _on_grid(field: xr.DataArray, like: xr.DataArray, what: str) -> xr.DataArray:
    """Select `field` at exactly `like`'s lat/lon (both 0.25 deg grids); a
    mismatch beyond 1e-6 deg is an error rather than a silent nearest pick."""
    out = field.sel(lat=like["lat"].values, lon=like["lon"].values,
                    method="nearest", tolerance=1e-6)
    return out.assign_coords(lat=like["lat"].values, lon=like["lon"].values)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_case(cfg: SeasonConfig, job: Job, truth: TruthCache,
               weights: xr.DataArray) -> xr.Dataset:
    fc = open_forecast(cfg, job)
    init = pd.Timestamp(job.init_time)
    valid = [init + pd.Timedelta(lt) for lt in fc["lead_time"].values]
    obs = truth.fields(valid).rename(time="lead_time").assign_coords(
        lead_time=fc["lead_time"].values)
    obs = _on_grid(obs, fc, truth.truth)
    w = _on_grid(weights, fc, "land weights")
    seed = zlib.crc32(f"{job.model}|{init.isoformat()}|{truth.truth}".encode())
    stats = scores.case_statistics(fc, obs, w, thresholds=None, seed=seed)
    stats = stats.expand_dims(init_time=[np.datetime64(init, "ns")])
    stats.attrs.update(model=job.model, truth=truth.truth, init_time=job.init_time,
                       forecast_written_utc=_manifest_time(job) or "",
                       n_members=int(fc.sizes["ensemble"]))
    return stats


def run_scoring(cfg: SeasonConfig, models: list[str] | None = None,
                truths: list[str] | None = None, force: bool = False,
                truth_sources: dict | None = None, mask_source=None) -> dict:
    """Score every finished forecast against every truth. Returns counts."""
    truths = truths or list(cfg.truths)
    unknown = set(truths) - set(cfg.truths)
    if unknown:
        raise ValueError(f"{sorted(unknown)} not among the config's truths {list(cfg.truths)}")
    weights = land_weights(cfg, mask_source)
    logger.info("land weights: %d of %d cells", int((weights > 0).sum()), weights.size)

    finished = [j for j in build_jobs(cfg, models) if is_done(j, cfg)]
    summary = {"scored": 0, "up_to_date": 0, "failed": []}
    for truth_name in truths:
        source = (truth_sources or {}).get(truth_name)
        truth = TruthCache(cfg, truth_name, source)
        for job in finished:
            path = stats_path(cfg, truth_name, job)
            if not force and path.exists():
                with xr.open_dataset(path) as old:
                    if old.attrs.get("forecast_written_utc") == (_manifest_time(job) or ""):
                        summary["up_to_date"] += 1
                        continue
            try:
                stats = score_case(cfg, job, truth, weights)
            except Exception as exc:          # one bad case must not stop the season
                logger.error("FAILED %s vs %s: %s", job.label, truth_name, exc)
                summary["failed"].append(f"{job.label} vs {truth_name}: {type(exc).__name__}: {exc}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            stats.to_netcdf(tmp)
            tmp.rename(path)
            summary["scored"] += 1
            logger.info("scored %s vs %s", job.label, truth_name)
    summary["finished_forecasts"] = len(finished)
    return summary


# ---------------------------------------------------------------------------
# loading for the report
# ---------------------------------------------------------------------------

def load_statistics(cfg: SeasonConfig, truth: str,
                    models: list[str] | None = None) -> dict[str, xr.Dataset]:
    """{model: Dataset with an init_time dim} of every scored case. Sum over
    init_time (or any subset of it), then scores.finalise."""
    out = {}
    for spec in cfg.models:
        if models and spec.name not in models:
            continue
        files = sorted((_root(cfg) / "scores" / truth).glob(f"{spec.name}_*.nc"))
        if not files:
            continue
        parts = [xr.open_dataset(f).load() for f in files]
        ds = xr.concat(parts, dim="init_time").sortby("init_time")
        ds.attrs = {"n_members": parts[0].attrs["n_members"], "model": spec.name, "truth": truth}
        out[spec.name] = ds
    return out


def season_scores(stats: xr.Dataset, init_times=None) -> xr.Dataset:
    """Finalised metrics over the given init times (default: all)."""
    sel = stats if init_times is None else stats.sel(init_time=init_times)
    total = sel.sum("init_time")
    total.attrs = stats.attrs
    return scores.finalise(total)
