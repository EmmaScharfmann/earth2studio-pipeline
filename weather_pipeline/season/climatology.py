"""
ERA5 climatology over the season's region: extreme thresholds and a
climatological ensemble baseline (Wessel et al., Sections 4.1 and 4.3).

Source: the Arraylake ERA5 repo's `single/temporal` group, the same ERA5
chunked along time (about a year per chunk, 12 x 12 cells), so a region's
40-year history is a few dozen chunk columns instead of 350,000 global
fields. 1979-2017 is final ERA5 there.

Cache, under <output_dir>/climatology/<region>_<variable>_<y0>-<y1>/:
    years/<year>.nc     the region at 00/06/12/18 UTC for one year
    thresholds.nc       (quantile, month, hour, lat, lon)
Years are fetched in parallel and each is written once, so an interrupted
build resumes where it stopped.

Thresholds: per grid cell, calendar month and hour of day, the empirical
quantiles of all samples in that month and hour over the years (about
39 x 30 values each); quantile 1.0 is the maximum, i.e. the record. Fixed in
advance from data that does not include the season, as the paper requires
so that scores conditioned on them are proper.

Climatological ensemble for a valid time: the value at the same calendar
day and hour in every climatology year (29 February falls back to the
28th), subsampled to the season's scored member count like the models.
"""

from __future__ import annotations

import logging
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import forecast_runner
from ..regions import crop_indices, parse_region
from .config import SeasonConfig

logger = logging.getLogger(__name__)

ARRAYLAKE_REPO = "earthmover-public/era5"
TEMPORAL_GROUP = "single/temporal"
SYNOPTIC_HOURS = (0, 6, 12, 18)
VARIABLE_NAMES = {"t2m": ["t2m", "2t", "2m_temperature"]}   # pipeline id -> names in the repo


def open_temporal() -> xr.Dataset:
    from arraylake import Client
    session = Client().get_repo(ARRAYLAKE_REPO).readonly_session("main")
    return xr.open_zarr(session.store, group=TEMPORAL_GROUP, chunks=None)


def _standard(ds: xr.Dataset) -> xr.Dataset:
    rename = {k: v for k, v in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")) if k in ds.dims}
    return ds.rename(rename)


class Climatology:
    def __init__(self, cfg: SeasonConfig, dataset: xr.Dataset | None = None):
        self.cfg = cfg
        self.years = range(cfg.climatology_years[0], cfg.climatology_years[1] + 1)
        tag = f"{cfg.region.replace(',', '_')}_{cfg.variable}_{self.years[0]}-{self.years[-1]}"
        self.dir = forecast_runner.REPO_ROOT / cfg.output_dir / "climatology" / tag
        self._dataset = dataset
        self._series: xr.DataArray | None = None

    # ---- raw data ---------------------------------------------------------

    @property
    def dataset(self) -> xr.Dataset:
        if self._dataset is None:
            logger.info("Opening %s (%s)", ARRAYLAKE_REPO, TEMPORAL_GROUP)
            self._dataset = open_temporal()
        return self._dataset

    def _field(self) -> xr.DataArray:
        ds = _standard(self.dataset)
        for name in VARIABLE_NAMES.get(self.cfg.variable, [self.cfg.variable]):
            if name in ds:
                return ds[name]
        raise KeyError(f"No {self.cfg.variable} in the climatology source; has {list(ds.data_vars)[:10]}...")

    def _year_path(self, year: int) -> Path:
        return self.dir / "years" / f"{year}.nc"

    def _fetch_year(self, year: int) -> Path:
        path = self._year_path(year)
        if path.exists():
            return path
        da = self._field()
        i, j = crop_indices(parse_region(self.cfg.region), da["lat"].values, da["lon"].values)
        times = pd.DatetimeIndex(da["time"].values)
        pick = np.flatnonzero((times.year == year) & np.isin(times.hour, SYNOPTIC_HOURS))
        if pick.size == 0:
            raise LookupError(f"No {year} data in the climatology source")
        out = da.isel(time=pick, lat=i, lon=j).load().astype("float32")
        out = out.drop_vars([c for c in out.coords if c not in ("time", "lat", "lon")], errors="ignore")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        out.rename(self.cfg.variable).to_netcdf(tmp)
        tmp.rename(path)
        logger.info("climatology: %d cached (%d times)", year, out.sizes["time"])
        return path

    def build(self, workers: int = 8) -> None:
        """Fetch every missing year (in parallel), then the thresholds."""
        missing = [y for y in self.years if not self._year_path(y).exists()]
        logger.info("climatology %s: %d of %d year(s) to fetch", self.dir.name, len(missing), len(self.years))
        if missing:
            with ThreadPoolExecutor(max_workers=max(1, min(workers, len(missing)))) as pool:
                list(pool.map(self._fetch_year, missing))
        self.thresholds()

    def is_built(self) -> bool:
        return (self.dir / "thresholds.nc").exists()

    @property
    def series(self) -> xr.DataArray:
        if self._series is None:
            missing = [y for y in self.years if not self._year_path(y).exists()]
            if missing:
                raise FileNotFoundError(f"climatology years not built: {missing[:5]}...; "
                                        "run `run_season.py <config> --stage climatology`")
            parts = [xr.open_dataarray(self._year_path(y)).load() for y in self.years]
            self._series = xr.concat(parts, dim="time")
        return self._series

    # ---- thresholds ---------------------------------------------------------

    def thresholds(self) -> xr.DataArray:
        """(quantile, month, hour, lat, lon), quantile 1.0 = record. Cached."""
        path = self.dir / "thresholds.nc"
        if path.exists():
            return xr.open_dataarray(path).load()
        s = self.series
        t = pd.DatetimeIndex(s["time"].values)
        q = np.array(self.cfg.quantiles, dtype=float)
        out = np.full((len(q), 12, len(SYNOPTIC_HOURS)) + s.shape[1:], np.nan, dtype="float32")
        for m in range(1, 13):
            for h, hour in enumerate(SYNOPTIC_HOURS):
                sel = (t.month == m) & (t.hour == hour)
                if sel.any():
                    out[:, m - 1, h] = np.quantile(s.values[sel], q, axis=0)
        th = xr.DataArray(out, dims=("quantile", "month", "hour", "lat", "lon"),
                          coords={"quantile": q, "month": np.arange(1, 13), "hour": list(SYNOPTIC_HOURS),
                                  "lat": s["lat"].values, "lon": s["lon"].values},
                          name="threshold",
                          attrs={"years": f"{self.years[0]}-{self.years[-1]}", "source": "ERA5"})
        path.parent.mkdir(parents=True, exist_ok=True)
        th.to_netcdf(path)
        return th

    def thresholds_for(self, valid_times) -> xr.DataArray:
        """(quantile, lead_time-like 'time', lat, lon) for the given valid times."""
        th = self.thresholds()
        t = pd.DatetimeIndex(pd.to_datetime(list(valid_times)))
        if not np.isin(t.hour, SYNOPTIC_HOURS).all():
            raise ValueError("thresholds exist for 00/06/12/18 UTC only")
        parts = [th.sel(month=v.month, hour=v.hour, drop=True) for v in t]
        return xr.concat(parts, dim="time").assign_coords(time=t.values).transpose("quantile", "time", "lat", "lon")

    # ---- climatological ensemble ---------------------------------------------

    def ensemble_for(self, valid_times, n_members: int | None = None, seed: int = 0) -> xr.DataArray:
        """(ensemble, time, lat, lon): the same calendar day and hour in each
        climatology year, subsampled reproducibly to n_members."""
        s = self.series
        index = pd.DatetimeIndex(s["time"].values)
        fields = []
        for v in pd.to_datetime(list(valid_times)):
            dates = []
            for y in self.years:
                day = min(v.day, 28) if (v.month == 2 and v.day == 29) else v.day
                dates.append(pd.Timestamp(year=y, month=v.month, day=day, hour=v.hour))
            pos = index.get_indexer(dates)
            if (pos < 0).any():
                raise LookupError(f"climatology lacks {np.array(dates)[pos < 0][:3]}")
            fields.append(s.isel(time=pos).rename(time="ensemble").assign_coords(ensemble=np.arange(len(dates))))
        ens = xr.concat(fields, dim="time").assign_coords(time=pd.to_datetime(list(valid_times)).values)
        if n_members is not None and n_members < ens.sizes["ensemble"]:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(ens.sizes["ensemble"], n_members, replace=False))
            ens = ens.isel(ensemble=keep).assign_coords(ensemble=np.arange(n_members))
        return ens.transpose("ensemble", "time", "lat", "lon")


def member_seed(init_time) -> int:
    return zlib.crc32(f"climatology|{pd.Timestamp(init_time).isoformat()}".encode())
