# earth2studio-pipeline

Run Aurora / AIFS / AIFS-ENS / Atlas on Arraylake ERA5 (or IFS) initial conditions via Earth2Studio,
with shared postprocessing, scoring against ERA5 and weather stations, and rotating-globe visualisation.

The downscaler that sits on top of these forecasts lives in its own project,
[weather-forecast-model-downscaler](https://github.com/EmmaScharfmann/weather-forecast-model-downscaler),
and reads AIFS-ENS forecasts from Earthmover rather than from this pipeline.

## Setup

The base environment holds data access, scoring and plotting only. Each model's torch and
Earth2Studio pins live in `requirements/<model>.txt` and are layered on top per run, so models
with conflicting pins never share an environment.

Python 3.12 is required (pinned in `.python-version`): the AIFS environments use flash-attn and torch wheels built for 3.12 only.

```bash
uv sync
uv run arraylake auth login        # or export ARRAYLAKE_TOKEN=ema_...
```

## Run

```bash
uv run scripts/run_benchmark.py --models aifs-ens1.0 --n-members 10 --source era5 \
    --init-times 2026-03-13T00:00:00 --n-steps 8 --variable t2m
```

`run_benchmark.py` runs each model in its own environment built from `requirements/<model>.txt`.

For long runs, store only what you score. `--store-variables` and `--store-region` crop the forecast zarr at write time for every model (a France-only t2m ensemble is a few MB per run instead of GBs):

```
uv run scripts/run_benchmark.py --models aifs-ens1.0 aurora1.5-ens-6h --n-members 10 \
    --init-times 2026-06-01T00:00:00 --n-steps 20 --variables t2m \
    --store-variables t2m --store-region france
```

Ensemble runs whose members come out identical are rejected before they are recorded as complete (`weather_pipeline/ensemble_checks.py`).

## Seasonal evaluation (in progress)

`weather_pipeline/season/` evaluates AI ensembles over a season, following the probabilistic protocol of Wessel et al. (arXiv:2610.20519). The period is an input (`configs/summer2026_france.yaml`: `period: {season: JJA, year: 2026}` or explicit dates). Truths are ERA5 and the step-0 HRES field from ECMWF open data (`weather_pipeline/ifs_opendata.py`, 2024 onward); `season/scores.py` holds the fair, additive scores.

```bash
uv run scripts/run_season.py configs/summer2026_france.yaml --stage status
uv run scripts/run_season.py configs/summer2026_france.yaml --stage forecast --limit 2   # a first test
uv run scripts/run_season.py configs/summer2026_france.yaml --stage forecast             # the rest; rerun to resume
```

Forecasts go to `outputs/season/<name>/forecasts/`, one store per (model, initialisation), separate from `run_benchmark.py`'s `outputs/forecasts/`. `--start/--end` or `--season/--year` run another period without editing the config; `--models` restricts to some models; `--dry-run` lists the jobs. Failed jobs are logged to `forecast_failures.json` and retried on the next run.

On a Slurm node with several GPUs, run one process per GPU, each with a cost-balanced share of the jobs (`--shard i --num-shards n`):

```bash
mkdir -p logs
sbatch scripts/slurm/season_forecast.sh configs/summer2026_france.yaml
```

Resubmitting the same command resumes. `--stage status --num-shards 8` shows how the jobs split across GPUs.

Each job is a separate `uv run`, so the checkpoint is loaded once per initialisation (about 35 s for AIFS-ENS, on top of the forecast itself).

## Contents

- `weather_pipeline/`: data sources (Arraylake ERA5, IFS), model backends, postprocessing, station
  and WeatherBench-X scoring, plotting.
- `scripts/`: `run_benchmark.py` (forecasts), `m0_scorecard.py` / `make_report.py` (scores),
  `fetch_stations.py`, `export_datapack.py`, `show_regions.py`, `regrid_diagnostic.py`.
- `tutorial.ipynb`: run Aurora 1.5 ENS from ERA5 and compare it with the ERA5 analysis.
- `experiments/`: exploratory notebooks.
- `findings.md`: data and library behaviour worth knowing, shared with the downscaler project.
