"""
A season config as a list of forecast jobs, one per (model, init time).

Each job has its own zarr store and manifest under
    <output_dir>/<config name>/forecasts/<model>_<YYYYmmddTHHMM>.zarr
so seasons never overwrite each other or the stores of run_benchmark.py
(outputs/forecasts/), and a job is "done" exactly when its manifest matches
what the config asks for (forecast_runner.existing_forecast_matches).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from ..forecast_runner import (BENCHMARK_STEP_HOURS, MODEL_STEP_HOURS,
                              existing_forecast_matches, init_slug)
from .config import SeasonConfig

INIT_FMT = "%Y-%m-%dT%H:%M:%S"     # the form run_benchmark passes to models


@dataclass(frozen=True)
class Job:
    model: str
    init_time: str            # "2026-06-01T00:00:00"
    n_members: int
    n_steps: int              # benchmark (6 h) steps, as run_benchmark counts
    out_zarr: str             # relative to the repo root

    @property
    def native_steps(self) -> int:
        """Steps of the model's own length. Manifests record THIS count
        (run_benchmark passes native steps to run_model_isolated), so it is
        also what decides whether a stored forecast matches."""
        step = MODEL_STEP_HOURS.get(self.model, BENCHMARK_STEP_HOURS)
        return self.n_steps * BENCHMARK_STEP_HOURS // step

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.init_time}"


def season_dir(cfg: SeasonConfig) -> str:
    return f"{cfg.output_dir}/{cfg.name}"


def store_path(cfg: SeasonConfig, model: str, init_time) -> str:
    return f"{season_dir(cfg)}/forecasts/{model}_{init_slug(str(init_time))}.zarr"


def build_jobs(cfg: SeasonConfig, models: list[str] | None = None) -> list[Job]:
    """All jobs of the season, initialisation-major: every model for the
    first init, then the next, so a partly finished season is a complete
    comparison over its first dates rather than one model's whole summer."""
    specs = [m for m in cfg.models if models is None or m.name in models]
    if models:
        unknown = set(models) - {m.name for m in cfg.models}
        if unknown:
            raise ValueError(f"{sorted(unknown)} not in the config's models "
                             f"{[m.name for m in cfg.models]}")
    jobs = []
    for init in cfg.init_times:
        init_str = pd.Timestamp(init).strftime(INIT_FMT)
        for spec in specs:
            jobs.append(Job(spec.name, init_str, spec.members, cfg.n_steps,
                            store_path(cfg, spec.name, init_str)))
    return jobs


def is_done(job: Job, cfg: SeasonConfig) -> bool:
    return existing_forecast_matches(
        job.model, job.init_time, job.native_steps, job.out_zarr, job.n_members,
        cfg.ic_source, [cfg.variable], cfg.region,
    )


def status(cfg: SeasonConfig, jobs: list[Job]) -> dict[str, Counter]:
    """{model: Counter(done=.., pending=..)}"""
    out: dict[str, Counter] = {}
    for job in jobs:
        out.setdefault(job.model, Counter())["done" if is_done(job, cfg) else "pending"] += 1
    return out
