"""
Check forecast stores written with --store-variables / --store-region.

Run after a smoke test such as:

    uv run scripts/run_benchmark.py \
        --models aifs-ens1.0 aurora1.5-ens-6h atlas-crps \
        --n-members 2 --init-times 2026-06-01T00:00:00 --n-steps 2 \
        --variables t2m --store-variables t2m --store-region france

then:

    uv run scripts/check_forecast_storage.py \
        --models aifs-ens1.0 aurora1.5-ens-6h atlas-crps \
        --init-time 2026-06-01T00:00:00 --members 2 --region france

For each model it checks that the store holds only the requested variables,
is cropped to the region, has the requested members, that those members
actually differ, and that the manifest records the selection. Exits non-zero
if anything fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from weather_pipeline.ensemble_checks import MIN_SPREAD, first_lead_spread  # noqa: E402
from weather_pipeline.regions import parse_region, to_signed_lon  # noqa: E402

GRID_STEP = 0.25


def store_path(model: str, init_time: str) -> Path:
    slug = pd.Timestamp(init_time).strftime("%Y%m%dT%H%M")
    return REPO_ROOT / f"outputs/forecasts/{model}_{slug}.zarr"


def disk_mb(path: Path) -> float:
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / 1024 if out.returncode == 0 else float("nan")


def check(model: str, init_time: str, members: int, variables: list[str],
          region: str) -> list[tuple[str, bool, str]]:
    path = store_path(model, init_time)
    if not path.exists():
        return [("store exists", False, f"{path} not found")]

    results = []
    ds = xr.open_zarr(path)
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"

    have = sorted(ds.data_vars)
    results.append(("only requested variables", have == sorted(variables), f"{have}"))

    lat0, lat1, lon0, lon1 = parse_region(region)
    lat = ds[lat_name].values
    lon = to_signed_lon(ds[lon_name].values)
    want_lat = round((lat1 - lat0) / GRID_STEP) + 1
    want_lon = round((lon1 - lon0) / GRID_STEP) + 1
    # a model on a 720-row grid (no south pole row) can miss one latitude
    lat_ok = want_lat - 1 <= lat.size <= want_lat
    results.append(("cropped latitudes", lat_ok and lat.min() >= lat0 and lat.max() <= lat1,
                    f"{lat.size} rows, {lat.max():.2f} to {lat.min():.2f}"))
    lon_ok = (lon.size == want_lon and np.isclose(lon[0], lon0) and np.isclose(lon[-1], lon1)
              and np.all(np.diff(lon) > 0))
    results.append(("cropped longitudes, west to east", lon_ok,
                    f"{lon.size} cols, {lon[0]:.2f} to {lon[-1]:.2f}"))

    n = ds.sizes.get("ensemble", 0)
    results.append(("members", n == members, f"{n} (want {members})"))

    if n >= 2:
        spread = first_lead_spread(ds, variables[0])
        results.append(("members differ", spread > MIN_SPREAD,
                        f"max spread at +6 h {spread:.3f} K"))

    hours = sorted({int(h) for h in ds["lead_time"].values / np.timedelta64(1, "h")})
    step = np.diff(hours).min() if len(hours) > 1 else 0
    results.append(("6-hourly lead times", step == 6,
                    f"step {step} h, leads {hours[0]}..{hours[-1]} h"))

    manifest = Path(str(path) + ".manifest.json")
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        ok = meta.get("store_variables") == variables and meta.get("store_region") == region
        results.append(("manifest records selection", ok,
                        f"{meta.get('store_variables')} / {meta.get('store_region')}"))
    else:
        results.append(("manifest records selection", False, "no manifest (run incomplete?)"))

    results.append(("size on disk", True, f"{disk_mb(path):.1f} MB"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--init-time", required=True)
    parser.add_argument("--members", type=int, required=True)
    parser.add_argument("--variables", nargs="+", default=["t2m"])
    parser.add_argument("--region", default="france")
    args = parser.parse_args()

    failed = 0
    for model in args.models:
        print(f"\n{model}")
        for name, ok, detail in check(model, args.init_time, args.members,
                                      args.variables, args.region):
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    print(f"\n{'All checks passed.' if not failed else f'{failed} check(s) failed.'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
