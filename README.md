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

## Contents

- `weather_pipeline/`: data sources (Arraylake ERA5, IFS), model backends, postprocessing, station
  and WeatherBench-X scoring, plotting.
- `scripts/`: `run_benchmark.py` (forecasts), `m0_scorecard.py` / `make_report.py` (scores),
  `fetch_stations.py`, `export_datapack.py`, `show_regions.py`, `regrid_diagnostic.py`.
- `tutorial.ipynb`: run Aurora 1.5 ENS from ERA5 and compare it with the ERA5 analysis.
- `experiments/`: exploratory notebooks.
- `findings.md`: data and library behaviour worth knowing, shared with the downscaler project.
