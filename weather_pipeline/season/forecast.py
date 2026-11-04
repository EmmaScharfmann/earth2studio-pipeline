"""
Forecast stage: run the season's pending jobs, and open finished stores in
the one layout the scorer expects.

Running
-------
`run_forecasts` runs each pending job through forecast_runner.run_model_isolated
(one fresh `uv run` per job, in the model's own environment). A failed job is
recorded and skipped; the season continues. Rerunning picks up exactly the
jobs without a matching manifest, so an interrupted season resumes where it
stopped.

Loading
-------
The models write different stores: AIFS starts at the first forecast step,
Earth2Studio models include the initial state at lead 0, Earth2Studio
encodes lead_time in hours (xarray then warns while decoding), and member
counts can differ. `open_forecast` hides all of that and returns

    DataArray (ensemble, lead_time, lat, lon), lead_time timedelta64[ns]

holding exactly the config's lead times, selected BY VALUE, and exactly
`cfg.n_members_scored` members, subsampled reproducibly when a store has
more.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import forecast_runner
from ..ensemble_checks import DegenerateEnsembleError
from .config import SeasonConfig
from .schedule import Job, build_jobs, is_done, season_dir, shard_jobs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def run_forecasts(cfg: SeasonConfig, models: list[str] | None = None,
                  limit: int | None = None, dry_run: bool = False,
                  shard: int | None = None, num_shards: int = 1) -> dict:
    """Run pending jobs. Returns {"ran": [...], "failed": [...], "skipped": n}.

    With `shard`/`num_shards`, only this shard's jobs (see shard_jobs), so
    one process per GPU can share a season without overlap."""
    jobs = build_jobs(cfg, models)
    if shard is not None:
        jobs = shard_jobs(jobs, shard, num_shards)
    pending = [j for j in jobs if not is_done(j, cfg)]
    skipped = len(jobs) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    logger.info("%s: %d job(s) done, %d to run%s", cfg.name, skipped, len(pending),
                " (dry run)" if dry_run else "")

    ran, failed = [], []
    for i, job in enumerate(pending, 1):
        if dry_run:
            logger.info("[%d/%d] would run %s -> %s", i, len(pending), job.label, job.out_zarr)
            continue
        logger.info("[%d/%d] %s", i, len(pending), job.label)
        Path(forecast_runner.REPO_ROOT / job.out_zarr).parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            forecast_runner.run_model_isolated(
                job.model, job.init_time, job.native_steps,
                job.out_zarr, job.n_members, cfg.ic_source, [cfg.variable], cfg.region,
            )
        except (subprocess.CalledProcessError, DegenerateEnsembleError, OSError) as exc:
            logger.error("FAILED %s: %s", job.label, exc)
            failed.append({"job": job.label, "error": f"{type(exc).__name__}: {exc}"})
            _write_failures(cfg, failed, shard)
            continue
        ran.append({"job": job.label, "seconds": round(time.monotonic() - started)})

    return {"ran": ran, "failed": failed, "skipped": skipped}


def _write_failures(cfg: SeasonConfig, failed: list[dict], shard: int | None = None) -> None:
    """One file per shard, so parallel processes never overwrite each other."""
    name = "forecast_failures.json" if shard is None else f"forecast_failures_shard{shard}.json"
    path = forecast_runner.REPO_ROOT / season_dir(cfg) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failed, indent=2))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _member_seed(model: str, init_time: str) -> int:
    """Stable across runs and machines (unlike hash()), distinct per case."""
    return zlib.crc32(f"{model}|{pd.Timestamp(init_time).isoformat()}".encode())


# Earth2Studio stores lead_time in hours; recent xarray warns while decoding
# it to seconds ("Following pandas, xarray only supports decoding to
# timedelta64 values with a resolution of 's', 'ms', 'us', or 'ns' ...").
# Harmless here: open_forecast normalises lead_time to ns.
LEAD_TIME_DECODE_WARNING = r".*timedelta64 values with a resolution.*"


def open_store(path) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=LEAD_TIME_DECODE_WARNING)
        return xr.open_zarr(path)


def open_forecast(cfg: SeasonConfig, job: Job) -> xr.DataArray:
    path = forecast_runner.REPO_ROOT / job.out_zarr
    ds = open_store(path)
    if cfg.variable not in ds:
        raise KeyError(f"{job.label}: no '{cfg.variable}' in {path} (has {list(ds.data_vars)})")
    da = ds[cfg.variable]

    rename = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in da.dims}
    da = da.rename(rename)
    if "time" in da.dims:
        if da.sizes["time"] != 1:
            raise ValueError(f"{job.label}: expected one init time, store has {da.sizes['time']}")
        da = da.isel(time=0, drop=True)
    if "ensemble" not in da.dims:
        raise ValueError(f"{job.label}: no 'ensemble' dimension; not an ensemble store")

    leads = pd.to_timedelta(da["lead_time"].values).astype("timedelta64[ns]")
    da = da.assign_coords(lead_time=leads)
    want = cfg.lead_times.astype("timedelta64[ns]")
    missing = want.difference(pd.TimedeltaIndex(leads))
    if len(missing):
        raise ValueError(f"{job.label}: store lacks lead time(s) "
                         f"{[int(m / pd.Timedelta(hours=1)) for m in missing]} h")
    da = da.sel(lead_time=want.values)

    n_have, n_want = da.sizes["ensemble"], cfg.n_members_scored
    if n_have < n_want:
        raise ValueError(f"{job.label}: {n_have} members, config scores {n_want}")
    if n_have > n_want:
        rng = np.random.default_rng(_member_seed(job.model, job.init_time))
        keep = np.sort(rng.choice(n_have, n_want, replace=False))
        da = da.isel(ensemble=keep)

    return da.transpose("ensemble", "lead_time", "lat", "lon")
