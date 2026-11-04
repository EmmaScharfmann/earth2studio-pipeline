"""
IFS "fc0" truth from ECMWF open data on AWS: the step-0 field of each HRES
run, i.e. the quantity the verification literature calls HRES-fc0 (the
0th hour of the HRES forecast from the 4D-Var analysis).

Why this source: WeatherBench2's hres_t0 archive stops at 2022, and the
repo's Arraylake IFS source is a rolling two-week window, so neither covers
a past season after 2022. The public bucket `ecmwf-forecasts` keeps every
00/06/12/18 UTC run from 2024 onward, at 0.25 deg.

Only the bytes of the requested field are downloaded: each GRIB file has a
JSON-lines `.index` listing every message's offset and length, so one
ranged GET fetches 2t alone (~0.5 MB instead of ~40 MB per file). Fetched
messages are cached on disk, so a season is downloaded once.

Same call interface as ArraylakeERA5Source (Earth2Studio's DataSource
protocol): `source(times, variables)` -> DataArray (time, variable, lat,
lon), latitude descending, longitude 0..360 ascending. That lets it drop
into wbx_scoring.fetch_targets unchanged.
"""

from __future__ import annotations

import json
import logging
import time as _time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

BUCKET_URL = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
FIRST_DATE = pd.Timestamp("2024-01-01")

# pipeline channel id -> (ECMWF param, levtype)
PARAMS: dict[str, tuple[str, str]] = {
    "t2m": ("2t", "sfc"),
    "d2m": ("2d", "sfc"),
    "msl": ("msl", "sfc"),
    "u10m": ("10u", "sfc"),
    "v10m": ("10v", "sfc"),
    "lsm": ("lsm", "sfc"),       # land-sea fraction, for the land mask
}


class TruthUnavailable(LookupError):
    """The requested valid time / field is not in the open-data archive."""


def file_url(valid_time, suffix: str = "grib2") -> str:
    """Step-0 HRES file for a valid time. The 06/18 UTC runs live under the
    same `oper` stream in the open-data layout (verified for 2026)."""
    t = pd.Timestamp(valid_time)
    if t.hour % 6 or t.minute or t.second:
        raise TruthUnavailable(f"{t}: HRES runs only at 00/06/12/18 UTC")
    return (f"{BUCKET_URL}/{t:%Y%m%d}/{t:%H}z/ifs/0p25/oper/"
            f"{t:%Y%m%d%H}0000-0h-oper-fc.{suffix}")


def find_message(index_text: str, param: str, levtype: str = "sfc") -> tuple[int, int]:
    """(offset, length) of the step-0 message for `param` in an .index file."""
    for line in index_text.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if (entry.get("param") == param and entry.get("levtype") == levtype
                and str(entry.get("step")) == "0"):
            return int(entry["_offset"]), int(entry["_length"])
    raise TruthUnavailable(f"No step-0 '{param}' ({levtype}) message in index")


def _get(url: str, byte_range: tuple[int, int] | None = None,
         retries: int = 3) -> bytes:
    request = urllib.request.Request(url)
    if byte_range is not None:
        start, length = byte_range
        request.add_header("Range", f"bytes={start}-{start + length - 1}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                raise TruthUnavailable(f"{url}: HTTP {exc.code}") from exc
            if attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        _time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _decode(grib_path: Path) -> xr.DataArray:
    """One GRIB message -> (lat, lon), latitude descending, lon 0..360."""
    ds = xr.open_dataset(grib_path, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    da = ds[next(iter(ds.data_vars))].load()
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    da = da.drop_vars([c for c in da.coords if c not in ("lat", "lon")])
    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    da = da.assign_coords(lon=da["lon"].values % 360).sortby("lon")
    return da


class IFSOpenDataFC0Source:
    """Step-0 HRES fields from ECMWF open data, cached locally."""

    def __init__(self, cache_dir: str | Path = "~/.cache/weather_pipeline/ifs_fc0"):
        self.cache_dir = Path(cache_dir).expanduser()

    def _cache_path(self, valid_time, param: str) -> Path:
        t = pd.Timestamp(valid_time)
        return self.cache_dir / param / f"{t:%Y%m%d%H}.grib2"

    def fetch_field(self, valid_time, variable: str) -> xr.DataArray:
        if variable not in PARAMS:
            raise TruthUnavailable(
                f"'{variable}' not mapped to an ECMWF param; known: {sorted(PARAMS)}")
        t = pd.Timestamp(valid_time)
        if t < FIRST_DATE:
            raise TruthUnavailable(f"{t}: open-data archive starts {FIRST_DATE.date()}")
        param, levtype = PARAMS[variable]
        path = self._cache_path(t, param)
        if not path.exists():
            index = _get(file_url(t, "index")).decode()
            offset, length = find_message(index, param, levtype)
            data = _get(file_url(t), (offset, length))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(path)          # never leave a half-written cache entry
        return _decode(path)

    def __call__(self, time, variable) -> xr.DataArray:
        times = list(pd.to_datetime(np.atleast_1d(time).tolist()))
        variables = [variable] if isinstance(variable, str) else list(variable)
        logger.info("IFS fc0: %d time(s) x %s", len(times), variables)
        fields = [
            xr.concat([self.fetch_field(t, v) for v in variables], dim="variable")
            for t in times
        ]
        da = xr.concat(fields, dim="time")
        return da.assign_coords(time=np.array(times, dtype="datetime64[ns]"),
                                variable=np.array(variables))
