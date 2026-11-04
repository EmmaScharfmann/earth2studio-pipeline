"""
Benchmark orchestrator. Runs one or more models, each in its OWN
uv-managed environment (so aurora's torch pin never has to coexist
in-process with aifs's), then scores every model's forecast against
ERA5 truth using the shared base env this script itself runs in.

This script never imports torch, earth2studio, or any model backend
directly -- it only shells out to `uv run` per model and then reads back
the zarr file each run produces. That's what keeps it runnable in the
lightweight base env (numpy/xarray/matplotlib only) regardless of which
models are in the --models list.

Usage:
    uv run scripts/run_benchmark.py \
        --models aurora aifs1.0 atlas \
        --init-time 2026-03-13T00:00:00 --n-steps 8 --variable t2m

Add a model later by adding requirements/<name>.txt and a
BACKEND_FACTORIES entry in weather_pipeline/backends.py -- nothing in this
script needs to change.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess

import numpy as np
import xarray as xr

from weather_pipeline.datasource import build_data_source
from weather_pipeline.ensemble_checks import DegenerateEnsembleError
from weather_pipeline import wbx_scoring as wbx

logger = logging.getLogger(__name__)

from weather_pipeline.forecast_runner import (
    REPO_ROOT,
    STOCHASTIC_MODELS,
    existing_forecast_matches,
    forecast_path,
    init_slug,
    native_steps,
    run_model_isolated,
)


def score_case(
    model: str, init_time: str, out_zarr: str, variables: list[str],
    data_source, metrics="all", stride_hours: int = 6,
    region: str = "global", bin_regions: bool = False,
    climatology: str | None = None,
) -> None:
    """Score one (model, init_time) with WeatherBench-X.

    Runs in the BASE environment -- WB-X needs only xarray/numpy/beam, no
    torch -- so it works regardless of which model produced the forecast.

    Writes two things per case:
      * an AggregationState (zarr). NOT finalised metric values: combining
        cases means summing states and computing metrics once at the end,
        and storing finished RMSEs would make that impossible afterwards.
      * a small 2-D mean-bias field, cached so the report can draw spatial
        panels without re-reading forecasts or re-fetching truth.

    Also writes a per-case CSV of finalised values, purely for eyeballing a
    single forecast -- the report ignores it and uses the states.
    """
    # Models differ in what they predict: Aurora 1 emits four surface
    # variables, AIFS a dozen, and only AIFS and Aurora 1.5 produce
    # precipitation. Score what this model has and say what it lacks,
    # rather than failing the whole case over one missing variable.
    available = set(xr.open_zarr(out_zarr).data_vars)
    wanted = [v for v in variables if v in available]
    missing = [v for v in variables if v not in available]
    if missing:
        logger.info("%s does not predict %s -- scoring %s", model, missing, wanted)
    if not wanted:
        logger.warning("%s predicts none of %s; nothing to score", model, variables)
        return

    forecast = wbx.open_forecast(out_zarr, wanted)
    forecast = subsample_leads_wbx(forecast, stride_hours)

    # Subset BEFORE fetching truth: targets are built on the forecast's own
    # coordinates, so restricting the forecast restricts the truth fetch
    # too -- a regional run reads only the grid points it scores.
    if region != "global":
        box = wbx.parse_region(region)
        forecast = wbx.subset_region(forecast, box)
        logger.info("Region '%s' %s -> %d x %d grid points", region, box,
                    forecast.sizes["latitude"], forecast.sizes["longitude"])

    # One truth fetch for every variable -- that is the expensive part, and
    # the alternative (a fetch per variable) would re-read the same valid
    # times repeatedly.
    targets = wbx.fetch_targets(data_source, forecast, wanted)
    ensemble = wbx.has_ensemble(forecast)

    slug = init_slug(init_time)
    scores_dir = REPO_ROOT / "outputs/scores"
    fields_dir = REPO_ROOT / "outputs/fields"
    scores_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)

    # A state PER VARIABLE, not one covering all of them: the report globs
    # by variable, and models cover different variable sets, so keeping
    # them separate means a model missing one variable still contributes to
    # the others.
    clim = wbx.load_climatology(climatology, wanted) if climatology else None
    # Binning computes EVERY region in one pass, adding a `region`
    # dimension -- unlike --region, which throws the rest of the globe away.
    # Kept under a separate tag because the resulting states have an extra
    # dimension the global report does not expect.
    bins = wbx.build_binning(regions=True) if bin_regions else None

    for variable in wanted:
        state, results, _ = wbx.score_case(
            forecast, targets, [variable], climatology=clim, bin_by=bins
        )
        df = wbx.to_dataframe(results)

        # Region goes in the filename so global and regional scores live
        # side by side and the report never mixes them by accident.
        tag = variable if region == "global" else f"{variable}@{region}"

        # Binned states go in their OWN directory rather than carrying a
        # marker in the filename: zarr/fsspec treat "#" as a URL fragment
        # separator and silently truncate the path at it, so a "#binned"
        # suffix produced files named for everything BEFORE the marker.
        out_dir = scores_dir / "binned" if bin_regions else scores_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        state_path = out_dir / f"{model}_{tag}_{slug}_state.zarr"
        shutil.rmtree(state_path, ignore_errors=True)
        # Record whether this was an ensemble: metric_values needs the same
        # metric objects the state was built with, and the report reads the
        # flag back rather than guessing from the stored statistic names.
        wbx.save_state(state, state_path, ensemble=ensemble)
        df.to_csv(out_dir / f"{model}_{tag}_{slug}_metrics.csv")

        # Bias map from the SAME arrays the metrics used -- no second load
        # of the forecast, no second truth fetch, and no risk of the map
        # covering different lead times or a different region.
        bias = wbx.mean_bias_field(forecast, targets, variable)
        bias_path = fields_dir / f"{model}_{tag}_{slug}_bias.zarr"
        shutil.rmtree(bias_path, ignore_errors=True)
        bias.to_dataset().to_zarr(bias_path, mode="w")

        logger.info("%s @ %s (%s):\n%s", model, init_time, variable, df)


def subsample_leads_wbx(forecast, stride_hours: int):
    """Same cadence filter as subsample_leads, on WB-X's layout where lead
    time is its own dimension rather than folded into valid_time."""
    hours = (forecast["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    # h=0 is the initial condition, not a forecast: Earth2Studio's iterator
    # yields it first, and scoring it gives a meaningless perfect zero.
    keep = (hours > 0) & (hours % stride_hours == 0)
    if keep.all():
        return forecast
    logger.info("Scoring %d of %d lead times (%d h cadence)",
                int(keep.sum()), len(hours), stride_hours)
    return forecast.isel(lead_time=np.flatnonzero(keep))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Run + score Earth2Studio/anemoi models, each isolated in its own uv env"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=["aurora", "aurora1.5", "aurora1.5-ens", "aurora1.5-ens-6h",
                 "atlas", "atlas-crps",
         "aifs1.0", "aifs1.1", "aifs-ens1.0"],
    )
    parser.add_argument(
        "--init-times", nargs="+", required=True,
        help="One or more init times, e.g. 2026-03-13T00:00:00 2026-03-14T00:00:00. "
             "Every model is run for every init time; scores are aggregated "
             "across them by scripts/make_report.py.",
    )
    parser.add_argument(
        "--n-steps", type=int, required=True,
        help="Forecast length in 6-HOUR steps, the same for every model. "
             "Models with a different native step (Aurora 1.5 runs hourly) "
             "are converted automatically, so --n-steps 8 is +48 h "
             "everywhere and the scores share a lead-time axis.",
    )
    parser.add_argument(
        "--n-members", type=int, default=None,
        help="Ensemble members for checkpoints that sample stochastically "
             "(aifs-ens*). Each member is a fresh draw from the same initial "
             "state, so this multiplies runtime by n_members. Deterministic "
             "models reject values > 1 rather than returning identical copies.",
    )
    parser.add_argument(
        "--variables", nargs="+", default=["t2m"],
        help="One or more variables, e.g. t2m z500 t850 msl. Each is scored "
             "and reported separately. Variables a model does not predict "
             "are skipped for that model with a log line -- coverage "
             "differs (only AIFS and Aurora 1.5 produce precipitation).",
    )
    parser.add_argument(
        "--region", default="global",
        help="Restrict scoring to a region: a name (global, europe, conus, "
             "tropics, arctic, antarctic, north-atlantic, nh-extratropics, "
             "sh-extratropics) or 'lat_min,lat_max,lon_min,lon_max' with "
             "longitudes in -180..180. Results are written under a separate "
             "tag so they never mix with global scores.",
    )
    parser.add_argument(
        "--bin-regions", action="store_true",
        help="Score every region in one pass (tropics, extratropics, "
             "arctic, antarctic, europe, north-america) instead of only "
             "globally. Results gain a 'region' dimension and are written "
             "under a '#binned' tag.",
    )
    parser.add_argument(
        "--climatology", default=None,
        help="Path/URL to a climatology zarr. Required for ACC and the "
             "activity diagnostic; without it those metrics are skipped.",
    )
    parser.add_argument(
        "--score-stride-hours", type=int, default=6,
        help="Score only lead times that are multiples of this many hours. "
             "Aurora 1.5 emits hourly lead times while the AIFS models emit "
             "6-hourly; 6 puts them on a common axis. Set to 1 to score every "
             "step a model produces (then models are NOT comparable).",
    )
    parser.add_argument(
        "--source", default="era5", choices=["era5", "ifs"],
        help="Initial conditions AND verification truth. era5: historical and "
             "reproducible but lags ~a quarter. ifs: rolling ~2-week window, "
             "and the analysis AIFS/Aurora1.5 were fine-tuned on -- recent "
             "dates only, and not reproducible once the window moves.",
    )
    parser.add_argument(
        "--metrics", default="all",
        help="Comma-separated metric names, or 'all' (default). Ensemble-only "
             "metrics are skipped automatically for deterministic forecasts "
             "under 'all', but named explicitly they raise. See --list-metrics.",
    )
    parser.add_argument("--list-metrics", action="store_true",
                        help="Print available metrics and exit.")
    parser.add_argument(
        "--store-variables", nargs="+", default=None, metavar="VAR",
        help="Write only these fields to the forecast zarr (e.g. t2m). "
             "Default: every field the model produces. Independent of "
             "--variables, which selects what is SCORED.",
    )
    parser.add_argument(
        "--store-region", default=None,
        help="Crop the forecast zarr at write time: a name from "
             "weather_pipeline.regions.REGIONS (e.g. france) or "
             "'lat_min,lat_max,lon_min,lon_max'. Default: global. A full "
             "season of global ensembles is TBs; a France t2m crop is GBs.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run models even when a matching forecast already exists.",
    )
    args = parser.parse_args()

    if args.list_metrics:
        print("Metrics come from WeatherBench-X; see "
              "weather_pipeline.wbx_scoring.build_metrics for the set.")
        return

    # Same source for truth as for initial conditions, so a run is
    # self-consistent. Scoring one analysis against another is a separate
    # (deliberate) experiment, not something to do by accident.
    data_source = build_data_source(args.source)

    failed = []
    for init_time in args.init_times:
        for model in args.models:
            out_zarr = forecast_path(model, init_time)
            label = f"{model} @ {init_time}"
            # --n-steps is in benchmark (6 h) steps; each model runs the
            # number of its OWN steps that reaches the same lead time.
            steps = native_steps(model, args.n_steps)
            try:
                members = args.n_members if model in STOCHASTIC_MODELS else None
                if not args.force and existing_forecast_matches(
                    model, init_time, steps, out_zarr, members, args.source,
                    args.store_variables, args.store_region,
                ):
                    logger.info("Reusing existing forecast for %s (use --force to re-run)", label)
                else:
                    run_model_isolated(model, init_time, steps, out_zarr,
                                       members, args.source,
                                       args.store_variables, args.store_region)
            except DegenerateEnsembleError:
                logger.exception("%s returned identical members -- not scored", label)
                failed.append(f"{label} (degenerate ensemble)")
                continue
            except subprocess.CalledProcessError:
                logger.exception("%s failed inside its own env -- skipping its scoring", label)
                failed.append(label)
                continue

            try:
                score_case(model, init_time, out_zarr, args.variables,
                           data_source, args.metrics, args.score_stride_hours,
                           args.region, args.bin_regions, args.climatology)
            except Exception:
                logger.exception("%s produced a forecast but scoring failed", label)
                failed.append(f"{label} (scoring)")

    if failed:
        logger.warning("Finished with failures: %s", failed)
    else:
        logger.info(
            "Complete: %d model(s) x %d init time(s). Build the comparison with:\n"
            "    uv run scripts/make_report.py --variable %s",
            len(args.models), len(args.init_times), args.variables[0],
        )


if __name__ == "__main__":
    main()