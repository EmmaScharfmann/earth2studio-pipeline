"""
A season config as a list of forecast jobs, one per (model, init time).

Each job has its own zarr store and manifest under
    <output_dir>/<config name>/forecasts/<model>_<YYYYmmddTHHMM>.zarr
so seasons never overwrite each other or the stores of run_benchmark.py
(outputs/forecasts/), and a job is "done" exactly when its manifest matches
what the config asks for (forecast_runner.existing_forecast_matches).
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from ..forecast_runner import (BENCHMARK_STEP_HOURS, MODEL_STEP_HOURS,
                              existing_forecast_matches, init_slug)
from .config import SeasonConfig

INIT_FMT = "%Y-%m-%dT%H:%M:%S"     # the form run_benchmark passes to models

# Minutes per job of 10 members to +120 h (H100), measured on the first run,
# scaled linearly with forecast length (job_minutes). Relative values are
# what balance the GPUs in shard_jobs; the absolute ones feed the estimates
# that run_season.py prints. A model missing here counts as
# DEFAULT_JOB_MINUTES; refine it once measured.
JOB_MINUTES = {
    "aifs-ens1.0": 2.3,
    "aurora1.5-ens-6h": 6.5,
}
DEFAULT_JOB_MINUTES = 5.0
_MEASURED_STEPS = 20            # the 120 h jobs the minutes were measured on


def job_minutes(job: "Job") -> float:
    per_120h = JOB_MINUTES.get(job.model, DEFAULT_JOB_MINUTES)
    return per_120h * job.n_steps / _MEASURED_STEPS


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


def shard_jobs(jobs: list[Job], shard: int, num_shards: int) -> list[Job]:
    """The jobs GPU `shard` (0-based) of `num_shards` should run.

    Longest-processing-time-first: jobs sorted by expected cost, each given
    to the least-loaded shard. Splitting every n-th job instead would hand
    whole models to single GPUs (the job list alternates models), leaving the
    fast models' GPUs idle while the slow ones run.

    Computed from the FULL job list, never the pending one, so a job always
    belongs to the same shard: rerunning the same array resumes each shard
    without two GPUs ever picking up the same forecast. Within a shard the
    original (initialisation-major) order is kept.
    """
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} outside 0..{num_shards - 1}")
    order = sorted(range(len(jobs)), key=lambda i: (-job_minutes(jobs[i]), i))
    heap = [(0.0, s) for s in range(num_shards)]
    owner = {}
    for i in order:
        load, s = heapq.heappop(heap)
        owner[i] = s
        heapq.heappush(heap, (load + job_minutes(jobs[i]), s))
    return [job for i, job in enumerate(jobs) if owner[i] == shard]


def expected_minutes(jobs: list[Job]) -> float:
    return sum(job_minutes(j) for j in jobs)


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
