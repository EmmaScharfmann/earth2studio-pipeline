import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # season_helpers

from season_helpers import write_store  # noqa: E402
from weather_pipeline import forecast_runner  # noqa: E402


@pytest.fixture
def fake_runner(monkeypatch):
    """Replace the GPU subprocess. Records calls; fails jobs listed in .fail."""
    calls, fail = [], set()

    def run(model, init_time, n_steps, out_zarr, n_members=None, source="era5",
            store_variables=None, store_region=None):
        calls.append((model, init_time, n_steps))
        if (model, init_time) in fail:
            raise subprocess.CalledProcessError(1, ["uv", "run"])
        write_store(forecast_runner.REPO_ROOT / out_zarr, model, n_members, n_steps * 6)
        forecast_runner.write_manifest(model, init_time, n_steps, out_zarr, n_members,
                                       source, store_variables, store_region)

    monkeypatch.setattr(forecast_runner, "run_model_isolated", run)
    run.calls, run.fail = calls, fail
    return run
