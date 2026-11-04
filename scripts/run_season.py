"""
Seasonal evaluation, stage by stage.

    uv run scripts/run_season.py configs/summer2026_france.yaml --stage status
    uv run scripts/run_season.py configs/summer2026_france.yaml --stage forecast

Stages
    status     how many forecasts are done / pending per model
    forecast   run the pending forecasts (GPU), resumable: rerun to continue
    score      score every finished forecast against each truth (CPU); only
               new or changed forecasts are scored, so rerun it any time

Useful options
    --models aifs-ens1.0         only these models (must be in the config)
    --limit 3                    run at most 3 pending jobs, e.g. a first test
    --dry-run                    list the jobs that would run, run nothing
    --truths ifs_fc0             score against these truths only
    --force                      rescore cases that are already scored
    --season JJA --year 2025     another period without editing the config
    --start 2026-07-01 --end 2026-07-07
    --shard 3 --num-shards 8     only this GPU's share of the jobs (balanced
                                 by expected cost; see scripts/slurm/)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_pipeline.season import config as season_config  # noqa: E402
from weather_pipeline.season.forecast import run_forecasts  # noqa: E402
from weather_pipeline.season.scoring import run_scoring  # noqa: E402
from weather_pipeline.season.schedule import (build_jobs, expected_minutes,  # noqa: E402
                                              shard_jobs, status)

logger = logging.getLogger("run_season")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seasonal evaluation of AI ensembles")
    parser.add_argument("config")
    parser.add_argument("--stage", choices=["status", "forecast", "score"], default="status")
    parser.add_argument("--truths", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard", type=int, default=None,
                        help="0-based index of this process's share of the jobs")
    parser.add_argument("--num-shards", type=int, default=1)
    period = parser.add_argument_group("period override")
    period.add_argument("--season", choices=sorted(season_config.SEASONS))
    period.add_argument("--year", type=int)
    period.add_argument("--start")
    period.add_argument("--end")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
                        datefmt="%H:%M:%S")

    overrides = {}
    if args.season or args.year:
        if not (args.season and args.year):
            parser.error("--season and --year go together")
        overrides["period"] = {"season": args.season, "year": args.year}
    if args.start or args.end:
        if overrides or not (args.start and args.end):
            parser.error("give --start and --end together, and not with --season")
        overrides["period"] = {"start": args.start, "end": args.end}

    cfg = season_config.load(args.config, **overrides)
    print(cfg.describe())

    if args.shard is not None and args.num_shards < 2:
        parser.error("--shard needs --num-shards >= 2")

    if args.stage == "status":
        jobs = build_jobs(cfg, args.models)
        counts = status(cfg, jobs)
        for model, c in counts.items():
            print(f"  {model:18s} {c['done']:4d} done  {c['pending']:4d} pending")
        if args.num_shards > 1:
            print(f"  balance over {args.num_shards} shards (expected minutes, all jobs):")
            for s in range(args.num_shards):
                mine = shard_jobs(jobs, s, args.num_shards)
                print(f"    shard {s}: {len(mine):3d} jobs, {expected_minutes(mine):6.0f} min")
        return 0

    if args.stage == "score":
        out = run_scoring(cfg, args.models, args.truths, args.force)
        print(f"\n{out['finished_forecasts']} finished forecast(s): scored {out['scored']}, "
              f"already up to date {out['up_to_date']}, failed {len(out['failed'])}")
        for f in out["failed"]:
            print(f"  FAILED {f}")
        return 1 if out["failed"] else 0

    summary = run_forecasts(cfg, args.models, args.limit, args.dry_run,
                            args.shard, args.num_shards)
    print(f"\nran {len(summary['ran'])}, failed {len(summary['failed'])}, "
          f"already done {summary['skipped']}")
    for r in summary["ran"]:
        print(f"  {r['job']}: {r['seconds'] / 60:.1f} min")
    if summary["ran"]:
        per_job = sum(r["seconds"] for r in summary["ran"]) / len(summary["ran"])
        jobs = build_jobs(cfg, args.models)
        if args.shard is not None:
            jobs = shard_jobs(jobs, args.shard, args.num_shards)
        remaining = sum(c["pending"] for c in status(cfg, jobs).values())
        scope = "" if args.shard is None else f" in shard {args.shard}"
        print(f"  ~{per_job / 60:.1f} min per job; {remaining} job(s) left{scope} "
              f"= ~{remaining * per_job / 3600:.1f} h at this rate")
    for f in summary["failed"]:
        print(f"  FAILED {f['job']}: {f['error']}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
