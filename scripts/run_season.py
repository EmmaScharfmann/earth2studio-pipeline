"""
Seasonal evaluation, stage by stage.

    uv run scripts/run_season.py configs/summer2026_france.yaml --stage status
    uv run scripts/run_season.py configs/summer2026_france.yaml --stage forecast

Stages
    status     how many forecasts are done / pending per model
    forecast   run the pending forecasts (GPU), resumable: rerun to continue

Useful options
    --models aifs-ens1.0         only these models (must be in the config)
    --limit 3                    run at most 3 pending jobs, e.g. a first test
    --dry-run                    list the jobs that would run, run nothing
    --season JJA --year 2025     another period without editing the config
    --start 2026-07-01 --end 2026-07-07
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_pipeline.season import config as season_config  # noqa: E402
from weather_pipeline.season.forecast import run_forecasts  # noqa: E402
from weather_pipeline.season.schedule import build_jobs, status  # noqa: E402

logger = logging.getLogger("run_season")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seasonal evaluation of AI ensembles")
    parser.add_argument("config")
    parser.add_argument("--stage", choices=["status", "forecast"], default="status")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
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

    if args.stage == "status":
        counts = status(cfg, build_jobs(cfg, args.models))
        for model, c in counts.items():
            print(f"  {model:18s} {c['done']:4d} done  {c['pending']:4d} pending")
        return 0

    summary = run_forecasts(cfg, args.models, args.limit, args.dry_run)
    print(f"\nran {len(summary['ran'])}, failed {len(summary['failed'])}, "
          f"already done {summary['skipped']}")
    for r in summary["ran"]:
        print(f"  {r['job']}: {r['seconds'] / 60:.1f} min")
    if summary["ran"]:
        per_job = sum(r["seconds"] for r in summary["ran"]) / len(summary["ran"])
        remaining = sum(c["pending"] for c in status(cfg, build_jobs(cfg, args.models)).values())
        print(f"  ~{per_job / 60:.1f} min per job; {remaining} job(s) left "
              f"= ~{remaining * per_job / 3600:.1f} h at this rate")
    for f in summary["failed"]:
        print(f"  FAILED {f['job']}: {f['error']}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
