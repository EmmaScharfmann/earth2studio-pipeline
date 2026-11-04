"""
Running one model forecast in its own uv environment, with manifests that
make runs resumable. Shared by scripts/run_benchmark.py (single cases) and
weather_pipeline/season (whole seasons).

Moved here unchanged from scripts/run_benchmark.py so both callers use the
same code; run_benchmark.py re-imports every name it used to define.

Base-environment module: no torch. Each forecast runs in a subprocess whose
environment is built from requirements/<model>.txt.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .ensemble_checks import check_member_spread

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent   # repo root, as before
REQUIREMENTS_DIR = REPO_ROOT / "requirements"

# Wheel indexes for the CUDA-specific builds the model envs need:
#   - torch's CUDA build (cu126) isn't on PyPI, only on pytorch's own index
#   - natten publishes prebuilt wheels per torch+CUDA combo on its own host;
#     without this it falls back to compiling from source, which appears to
#     hang for a very long time and needs nvcc/CUDA headers to even succeed
#
# Passed as `uv run` CLI FLAGS rather than set in pyproject.toml's [tool.uv]
# table, because config-file settings proved unreliable for the nested
# `--with-requirements` resolutions below -- honored on some runs, silently
# ignored on others, producing a confusing "no version of torch==2.13.0+cu126"
# even though that wheel demonstrably exists on the index. CLI flags are what
# uv's own error message points at, and they take precedence over config.
# Hardcoded here so nobody running the benchmark has to know or type them.
#
# If this repo is ever used on a machine with a different CUDA version, these
# URLs and the matching torch/natten pins in requirements/*.txt need to change
# together.
UV_INDEX_ARGS = [
    "--extra-index-url", "https://download.pytorch.org/whl/cu126",
    "--find-links", "https://whl.natten.org",
    # pytorch's index also mirrors common packages (certifi,
    # importlib-metadata, ...) at stale versions. uv's default is to take each
    # package from the first index that has it, which let those stale copies
    # win and made nvidia-physicsnemo's real requirements look unsatisfiable.
    # This makes uv compare versions across all indexes per package instead.
    "--index-strategy", "unsafe-best-match",
]

# Per-model extra `uv run` flags, for packages that ship a precompiled
# extension built against a different torch than the one we pin.
#
# torch-harmonics (atlas): its wheel bundles a compiled `_C` extension. If
# that was built against a different torch ABI than ours, importing it dies
# with "undefined symbol: _ZNR5torch7Library4_def...". Forcing a source
# build makes it compile against the torch actually installed in this env;
# --no-build-isolation-package lets the build see that torch (it needs
# torch's headers at build time, which an isolated build env wouldn't have).
#
# Expect this build to be slow. If it fails for lack of nvcc/CUDA headers,
# the fallback is a persistent venv for atlas rather than the ephemeral
# --with-requirements path.
# AIFS no longer needs any build flags: requirements/aifs*.txt pins the
# official prebuilt flash-attn wheel by URL, so nothing compiles. (Kept the
# lookup prefix-aware anyway -- if a future model does need build flags,
# this is where they go.)
# Models whose checkpoints sample stochastically, so repeated runs from the
# same initial condition give independent members and --n-members is
# meaningful. Listed explicitly rather than inferred from the name: Atlas
# is generative (stochastic interpolant) despite having no "-ens" suffix,
# and a deterministic model asked for members would silently return
# identical copies with zero spread.
#
# run_benchmark cannot import backends to ask (that would pull in torch),
# so this list has to be kept in step with backends.BACKEND_FACTORIES.
STOCHASTIC_MODELS = {"aifs-ens1.0", "aurora1.5-ens", "aurora1.5-ens-6h",
                     "atlas", "atlas-crps"}

# Each model's OUTPUT step in hours. Earth2Studio's hourly Aurora 1.5
# wrappers query the 6-hour autoregressive step at t+1h..t+6h before
# advancing the state. Each of those queries is a separate forward pass
# (Earth2Studio's _forward_sub_steps loops over the lead hours), so hourly
# output costs SIX model evaluations per 6 h cycle, not one. The *-6h
# variants query only t+6h. Everything else steps 6-hourly.
MODEL_STEP_HOURS = {
    "aurora": 6, "atlas": 6, "atlas-crps": 6,
    "aifs1.0": 6, "aifs1.1": 6, "aifs-ens1.0": 6,
    "aurora1.5": 1, "aurora1.5-ens": 1, "aurora1.5-ens-6h": 6,
}

# --n-steps counts steps of THIS cadence, for every model. Without the
# conversion below, --n-steps 8 means 48 h for AIFS but only 8 h for
# Aurora 1.5, and the scores end up on incomparable lead-time axes.
BENCHMARK_STEP_HOURS = 6


def native_steps(model: str, n_steps: int) -> int:
    """Convert benchmark steps (6 h each) into the model's own steps."""
    step = MODEL_STEP_HOURS.get(model, BENCHMARK_STEP_HOURS)
    steps = n_steps * BENCHMARK_STEP_HOURS // step
    if step != BENCHMARK_STEP_HOURS:
        logger.info(
            "%s steps every %d h: %d benchmark step(s) (%d h) -> %d native step(s)",
            model, step, n_steps, n_steps * BENCHMARK_STEP_HOURS, steps,
        )
    return steps

EXTRA_UV_ARGS: dict[str, list[str]] = {
    # Both Atlas variants need the torch-harmonics source build.
    "atlas": [
        "--no-binary-package", "torch-harmonics",
        "--no-build-isolation-package", "torch-harmonics",
    ],
}


def _extra_uv_args(model: str) -> list[str]:
    """Exact model name first, then prefix -- so every aifs* variant picks
    up the shared flash-attn handling without repeating it per version."""
    if model in EXTRA_UV_ARGS:
        return EXTRA_UV_ARGS[model]
    for prefix, args in EXTRA_UV_ARGS.items():
        if model.startswith(prefix):
            return args
    return []


def init_slug(init_time: str) -> str:
    """Filesystem-safe form of an init time, e.g. 2026-03-13T00:00:00 ->
    20260313T0000. Forecast paths include it so multiple init times coexist
    instead of overwriting each other."""
    return pd.Timestamp(init_time).strftime("%Y%m%dT%H%M")


def forecast_path(model: str, init_time: str) -> str:
    return f"outputs/forecasts/{model}_{init_slug(init_time)}.zarr"


def _manifest_path(out_zarr: str) -> Path:
    """Sidecar describing what produced a forecast zarr.

    A sidecar rather than zarr attrs because the two write paths differ:
    Earth2Studio's ZarrBackend owns the store for aurora/atlas, while the
    anemoi backend writes its own. A file next to the store works for both
    without either backend needing to cooperate."""
    return REPO_ROOT / (out_zarr + ".manifest.json")


def existing_forecast_matches(model: str, init_time: str, n_steps: int, out_zarr: str,
                              n_members: int | None = None, source: str = "era5",
                              store_variables: list[str] | None = None,
                              store_region: str | None = None) -> bool:
    """True if a usable forecast for exactly this request is already on disk.

    Checks the manifest rather than mere existence of the zarr: a store left
    behind by a different init time, a shorter run, or a crashed run
    mid-write would otherwise be silently reused and scored as if it were
    the requested forecast."""
    store = REPO_ROOT / out_zarr
    manifest = _manifest_path(out_zarr)
    if not store.exists() or not manifest.exists():
        return False
    try:
        meta = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Unreadable manifest at %s; re-running", manifest)
        return False

    # store_variables / store_region are part of the identity: a France-only
    # t2m store must never be reused for a global or multi-variable request.
    want = {"model": model, "init_time": init_time, "n_steps": n_steps,
            "n_members": n_members, "source": source,
            "store_variables": store_variables, "store_region": store_region}
    got = {k: meta.get(k) for k in want}
    if got != want:
        logger.info("Existing forecast at %s is for %s, want %s; re-running",
                    out_zarr, got, want)
        return False
    return True


def write_manifest(model: str, init_time: str, n_steps: int, out_zarr: str,
                   n_members: int | None = None, source: str = "era5",
                   store_variables: list[str] | None = None,
                   store_region: str | None = None) -> None:
    """Written only AFTER the run succeeds, so a crashed run leaves no
    manifest and is never mistaken for a complete forecast."""
    _manifest_path(out_zarr).write_text(json.dumps({
        "model": model,
        "init_time": init_time,
        "n_steps": n_steps,
        "n_members": n_members,
        "source": source,
        "store_variables": store_variables,
        "store_region": store_region,
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
    }, indent=2))


def run_model_isolated(model: str, init_time: str, n_steps: int, out_zarr: str,
                       n_members: int | None = None, source: str = "era5",
                       store_variables: list[str] | None = None,
                       store_region: str | None = None) -> None:
    """Runs one model's forecast to completion in its own uv environment.

    Each call is a fresh `uv run` invocation: uv resolves & (on first run)
    installs the base project deps plus this model's requirements file
    into an isolated environment, then executes the pipeline inside it.
    Later runs reuse uv's cache, so this is only slow the first time per
    model.

    The CUDA wheel indexes (UV_INDEX_ARGS above) are baked into the command
    so callers never have to set env vars or pass flags themselves."""
    req_file = REQUIREMENTS_DIR / f"{model}.txt"
    if not req_file.exists():
        raise FileNotFoundError(
            f"No requirements file for model '{model}': {req_file}. "
            "Add one alongside the others in requirements/ first."
        )

    cmd = [
        "uv", "run",
        *UV_INDEX_ARGS,
        *_extra_uv_args(model),
        "--with-requirements", str(req_file),
        "-m", "weather_pipeline.pipeline",
        "--model", model,
        "--init-time", init_time,
        "--n-steps", str(n_steps),
        "--out", out_zarr,
    ]
    if n_members is not None:
        cmd += ["--n-members", str(n_members)]
    cmd += ["--source", source]
    if store_variables:
        cmd += ["--variables", *store_variables]
    if store_region:
        cmd += ["--region", store_region]
    # Clear any existing store first. Earth2Studio's ZarrBackend OPENS an
    # existing zarr rather than replacing it, so a store left by a previous
    # run with a different number of steps keeps its old lead_time axis and
    # the run dies partway through with
    #   "Attempting to set a selection of 1 elements with an array of N"
    # at exactly the step where the old axis ran out.
    store = REPO_ROOT / out_zarr
    if store.exists():
        logger.info("Removing existing forecast store %s before re-running", store)
        shutil.rmtree(store, ignore_errors=True)
    manifest = _manifest_path(out_zarr)
    if manifest.exists():
        manifest.unlink()

    logger.info("Running '%s' in its own uv env: %s", model, " ".join(cmd))
    env = dict(os.environ)
    if model.startswith("atlas"):
        # Atlas asks for TF32 cuBLAS for inference performance -- it warns
        # on every load without it, and NVIDIA's own examples set it.
        env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    if model.startswith("atlas"):
        # Every Atlas variant gets the torch-harmonics source build (the
        # prefix match in _extra_uv_args), so every variant needs this too.
        # It used to be set for "atlas" only, so atlas-crps compiled a
        # CPU-only extension. uv's build cache does not key on this variable:
        # after changing it, `uv cache clean torch-harmonics` to force a
        # rebuild.
        #
        # torch-harmonics only compiles its CUDA kernels when this is set;
        # earth2studio's own install docs use it for the same package.
        # Without it the source build produces a CPU-only extension, which
        # imports fine but is much slower.
        env["FORCE_CUDA_EXTENSION"] = "1"
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)

    # An ensemble whose members are copies would score as a deterministic
    # forecast while labelled as an ensemble. Check before the manifest is
    # written, so a degenerate run is never treated as complete or reused.
    if n_members is not None and n_members > 1:
        spread = check_member_spread(str(store))
        logger.info("%s: max member spread at first lead = %.3f", model, spread)

    write_manifest(model, init_time, n_steps, out_zarr, n_members, source,
                   store_variables, store_region)
