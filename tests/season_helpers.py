"""Shared helpers for the season tests: a small config and stores shaped
like the ones the real backends write."""

import numpy as np
import xarray as xr

from weather_pipeline.season import config as C

LAT = np.arange(53, 39.75, -0.25)
LON = np.concatenate([np.arange(353, 360, 0.25), np.arange(0, 11.25, 0.25)])


def make_cfg(tmp_path, members=(4, 3), days=3, **kw):
    raw = {
        "name": "test",
        "period": {"start": "2026-07-01", "end": f"2026-07-{days:02d}"},
        "lead_hours": {"every": 6, "max": 24},
        "models": {"aifs-ens1.0": {"members": members[0]},
                   "aurora1.5-ens-6h": {"members": members[1]}},
        "output_dir": str(tmp_path),
        "truths": ["era5"],
        **kw,
    }
    return C.from_dict(raw)


def write_store(path, model, n_members, max_lead_h=24, seed=0):
    """A store as the given backend writes it."""
    g = np.random.default_rng(seed)
    anemoi = model.startswith("aifs")
    leads_h = np.arange(6 if anemoi else 0, max_lead_h + 1, 6)
    data = 290 + g.normal(size=(n_members, 1, len(leads_h), LAT.size, LON.size)).astype("f4")
    ds = xr.Dataset(
        {"t2m": (("ensemble", "time", "lead_time", "lat", "lon"), data)},
        coords={"ensemble": np.arange(n_members), "time": [np.datetime64("2026-07-01")],
                "lead_time": leads_h * np.timedelta64(1, "h"), "lat": LAT, "lon": LON},
    )
    # Earth2Studio encodes lead_time in hours, which makes xarray warn on read
    enc = {} if anemoi else {"lead_time": {"units": "hours", "dtype": "int64"}}
    ds.to_zarr(path, mode="w", encoding=enc)
