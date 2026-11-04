import numpy as np
import pytest
import xarray as xr

from weather_pipeline.ensemble_checks import (
    DegenerateEnsembleError, check_member_spread, first_lead_spread,
)


def forecast(spread_k: float, n_members=4, n_leads=3, seed=0):
    rng = np.random.default_rng(seed)
    base = 290 + rng.normal(size=(1, n_leads, 5, 6))
    noise = spread_k * rng.normal(size=(n_members, 1, n_leads, 5, 6))
    data = base[None] + noise
    data[:, :, 0] = base[:, 0]                 # lead 0 = shared initial state
    return xr.Dataset(
        {"t2m": (("ensemble", "time", "lead_time", "lat", "lon"), data)},
        coords={"ensemble": np.arange(n_members),
                "lead_time": np.arange(n_leads) * np.timedelta64(6, "h")},
    )


def test_lead_zero_is_ignored():
    # members identical at lead 0 by construction, different afterwards
    assert first_lead_spread(forecast(0.5)) > 0.1


def test_identical_members_are_rejected(tmp_path):
    path = tmp_path / "copies.zarr"
    forecast(0.0).to_zarr(path)
    with pytest.raises(DegenerateEnsembleError, match="identical"):
        check_member_spread(str(path))


def test_real_ensemble_passes(tmp_path):
    path = tmp_path / "ens.zarr"
    forecast(0.5).to_zarr(path)
    assert check_member_spread(str(path)) > 0.1


def test_single_member_is_not_an_ensemble():
    with pytest.raises(ValueError, match="two members"):
        first_lead_spread(forecast(0.5, n_members=1))
