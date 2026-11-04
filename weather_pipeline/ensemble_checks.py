"""
Sanity checks on ensemble forecasts. numpy/xarray only, so they run in the
base environment right after a model's own environment has written a zarr.

The failure this guards against is silent: a checkpoint or code path that
returns N identical "members" still produces a valid-looking ensemble zarr,
and every downstream score (CRPS, spread-skill, rank histograms) happily
computes on it. A zero-spread ensemble scores as a deterministic forecast
while being labelled as an ensemble, which would invalidate the comparison
the seasonal benchmark exists to make.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

ENSEMBLE_DIM = "ensemble"

# GPU non-determinism between identical runs is ~1e-6 K; genuine sampling
# differs by tenths of a kelvin already at the first step. Anything below
# this is treated as "the members are copies".
MIN_SPREAD = 1e-3


class DegenerateEnsembleError(RuntimeError):
    """An ensemble whose members are (numerically) identical."""


def first_lead_spread(ds: xr.Dataset, variable: str | None = None) -> float:
    """Largest across-member standard deviation at the first FORECAST lead.

    Lead zero is skipped: it is the shared initial condition, so its spread
    is zero by construction for every ensemble here.
    """
    if ENSEMBLE_DIM not in ds.dims:
        raise ValueError(f"No '{ENSEMBLE_DIM}' dimension; not an ensemble forecast")
    if ds.sizes[ENSEMBLE_DIM] < 2:
        raise ValueError("Spread needs at least two members")

    variable = variable or next(iter(ds.data_vars))
    da = ds[variable]

    if "lead_time" in da.dims:
        hours = da["lead_time"].values / np.timedelta64(1, "h")
        positive = np.flatnonzero(hours > 0)
        if positive.size == 0:
            raise ValueError("No lead time after the initial condition")
        da = da.isel(lead_time=int(positive[np.argmin(hours[positive])]))

    return float(da.std(dim=ENSEMBLE_DIM).max())


def check_member_spread(zarr_path: str, variable: str | None = None,
                        min_spread: float = MIN_SPREAD) -> float:
    """Raise DegenerateEnsembleError if the members do not differ."""
    ds = xr.open_zarr(zarr_path)
    spread = first_lead_spread(ds, variable)
    if not spread > min_spread:
        raise DegenerateEnsembleError(
            f"{zarr_path}: members are identical at the first lead time "
            f"(max spread {spread:.2e} <= {min_spread:.0e}). The model is not "
            "sampling, so this is not an ensemble."
        )
    return spread
