"""Forecast stage: scheduling, resuming, failure handling and loading.
The GPU subprocess is replaced by a fake that writes stores shaped like the
real ones (AIFS: no lead 0; Earth2Studio: lead 0, hour-encoded lead_time)."""


import numpy as np
import pytest

from weather_pipeline import forecast_runner
from weather_pipeline.season import config as C
from weather_pipeline.season import forecast as F
from weather_pipeline.season.schedule import build_jobs, is_done, status

from season_helpers import make_cfg, write_store

# ---- scheduling -----------------------------------------------------------

def test_jobs_are_initialisation_major(tmp_path):
    jobs = build_jobs(make_cfg(tmp_path))
    assert len(jobs) == 6
    assert [(j.model, j.init_time[:10]) for j in jobs[:3]] == [
        ("aifs-ens1.0", "2026-07-01"), ("aurora1.5-ens-6h", "2026-07-01"),
        ("aifs-ens1.0", "2026-07-02")]
    assert jobs[0].out_zarr == f"{tmp_path}/test/forecasts/aifs-ens1.0_20260701T0000.zarr"
    assert jobs[0].n_steps == 4 and jobs[0].native_steps == 4


def test_model_filter(tmp_path):
    assert {j.model for j in build_jobs(make_cfg(tmp_path), ["aifs-ens1.0"])} == {"aifs-ens1.0"}
    with pytest.raises(ValueError, match="not in the config"):
        build_jobs(make_cfg(tmp_path), ["atlas-crps"])


def test_hourly_model_counts_native_steps(tmp_path):
    cfg = C.from_dict({"name": "h", "period": {"start": "2026-07-01", "end": "2026-07-01"},
                       "lead_hours": {"max": 24}, "truths": ["era5"],
                       "models": {"aurora1.5-ens-6h": {"members": 2}}, "output_dir": str(tmp_path)})
    job = build_jobs(cfg)[0]
    object.__setattr__(job, "model", "aurora1.5-ens")          # hourly variant
    assert job.native_steps == 24


# ---- running & resuming -----------------------------------------------------

def test_run_then_resume(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path)
    first = F.run_forecasts(cfg, limit=2)
    assert len(first["ran"]) == 2 and first["skipped"] == 0
    counts = status(cfg, build_jobs(cfg))
    assert counts["aifs-ens1.0"] == {"done": 1, "pending": 2}

    second = F.run_forecasts(cfg)
    assert second["skipped"] == 2 and len(second["ran"]) == 4
    assert len(fake_runner.calls) == 6                         # nothing ran twice
    assert F.run_forecasts(cfg)["ran"] == []                   # all done


def test_failures_are_recorded_and_retried(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path)
    fake_runner.fail.add(("aurora1.5-ens-6h", "2026-07-02T00:00:00"))
    out = F.run_forecasts(cfg)
    assert len(out["ran"]) == 5 and len(out["failed"]) == 1
    assert (tmp_path / "test" / "forecast_failures.json").exists()

    fake_runner.fail.clear()
    retry = F.run_forecasts(cfg)
    assert [r["job"] for r in retry["ran"]] == ["aurora1.5-ens-6h @ 2026-07-02T00:00:00"]


def test_dry_run_runs_nothing(tmp_path, fake_runner):
    assert F.run_forecasts(make_cfg(tmp_path), dry_run=True)["ran"] == []
    assert fake_runner.calls == []


def test_store_for_another_config_is_not_reused(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path)
    F.run_forecasts(cfg, limit=1)
    longer = make_cfg(tmp_path, lead_hours={"every": 6, "max": 48})
    assert not is_done(build_jobs(longer)[0], longer)


# ---- loading ------------------------------------------------------------------

def test_lead_time_decode_warning_is_filtered():
    """The exact message from a real Atlas store (xarray on the GPU machine).
    Local xarray versions may not emit it, so the filter is tested directly."""
    import re
    real = ("Following pandas, xarray only supports decoding to timedelta64 values "
            "with a resolution of 's', 'ms', 'us', or 'ns'. Encoded values for "
            "variable 'lead_time' have a resolution of 'h'.")
    assert re.match(F.LEAD_TIME_DECODE_WARNING, real)


def test_open_forecast_normalises_both_store_layouts(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path, members=(4, 3), common_members=3)
    F.run_forecasts(cfg)
    for job in build_jobs(cfg)[:2]:
        da = F.open_forecast(cfg, job)
        assert da.dims == ("ensemble", "lead_time", "lat", "lon")
        assert da.sizes["ensemble"] == 3
        assert da["lead_time"].dtype == "timedelta64[ns]"
        assert list(da["lead_time"].values / np.timedelta64(1, "h")) == [6, 12, 18, 24]


def test_member_subsample_is_reproducible_and_case_specific(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path, members=(6, 3), common_members=3)
    F.run_forecasts(cfg)
    aifs = [j for j in build_jobs(cfg) if j.model == "aifs-ens1.0"]
    a1 = F.open_forecast(cfg, aifs[0])["ensemble"].values
    assert list(a1) == list(F.open_forecast(cfg, aifs[0])["ensemble"].values)
    picks = {tuple(F.open_forecast(cfg, j)["ensemble"].values) for j in aifs}
    assert len(picks) > 1                     # not the same members every day


def test_missing_lead_time_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path)
    job = build_jobs(cfg)[0]
    path = forecast_runner.REPO_ROOT / job.out_zarr
    path.parent.mkdir(parents=True)
    write_store(path, job.model, 4, max_lead_h=12)
    with pytest.raises(ValueError, match=r"lacks lead time\(s\) \[18, 24\]"):
        F.open_forecast(cfg, job)


def test_too_few_members_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path, members=(4, 3))
    job = build_jobs(cfg)[0]
    path = forecast_runner.REPO_ROOT / job.out_zarr
    path.parent.mkdir(parents=True)
    write_store(path, job.model, 2)
    with pytest.raises(ValueError, match="2 members, config scores 3"):
        F.open_forecast(cfg, job)


# ---- sharding across GPUs ---------------------------------------------------

from weather_pipeline.season import schedule as Sch  # noqa: E402


def season_jobs(tmp_path):
    cfg = C.from_dict({
        "name": "s", "period": {"season": "JJA", "year": 2026}, "truths": ["era5"],
        "models": {"aifs-ens1.0": {"members": 10}, "aurora1.5-ens-6h": {"members": 10}},
        "output_dir": str(tmp_path)})
    return cfg, Sch.build_jobs(cfg)


def test_shards_cover_every_job_exactly_once(tmp_path):
    _, jobs = season_jobs(tmp_path)
    shards = [Sch.shard_jobs(jobs, s, 8) for s in range(8)]
    labels = [j.label for sh in shards for j in sh]
    assert sorted(labels) == sorted(j.label for j in jobs)
    assert len(labels) == len(set(labels))


def test_shards_are_balanced_by_cost_not_count(tmp_path):
    _, jobs = season_jobs(tmp_path)
    loads = [Sch.expected_minutes(Sch.shard_jobs(jobs, s, 8)) for s in range(8)]
    # within one Aurora job of each other (LPT bound), and close to the mean
    assert max(loads) - min(loads) <= Sch.JOB_MINUTES["aurora1.5-ens-6h"]
    assert max(loads) < 1.05 * sum(loads) / 8
    # every GPU gets both models, rather than one model per GPU
    assert all(len({j.model for j in Sch.shard_jobs(jobs, s, 8)}) == 2 for s in range(8))


def test_shard_membership_does_not_depend_on_progress(tmp_path, fake_runner):
    cfg, jobs = season_jobs(tmp_path)
    before = [j.label for j in Sch.shard_jobs(jobs, 3, 8)]
    F.run_forecasts(cfg, limit=5)                      # some jobs now done
    after = [j.label for j in Sch.shard_jobs(Sch.build_jobs(cfg), 3, 8)]
    assert before == after


def test_shard_runs_only_its_jobs_and_logs_failures_separately(tmp_path, fake_runner):
    cfg = make_cfg(tmp_path, days=4)
    mine = {j.label for j in Sch.shard_jobs(build_jobs(cfg), 1, 2)}
    first = sorted(mine)[0]
    fake_runner.fail.add(tuple(first.split(" @ ")))
    out = F.run_forecasts(cfg, shard=1, num_shards=2)
    assert {r["job"] for r in out["ran"]} | {f["job"] for f in out["failed"]} == mine
    assert (tmp_path / "test" / "forecast_failures_shard1.json").exists()
    assert not (tmp_path / "test" / "forecast_failures.json").exists()


def test_bad_shard_index():
    with pytest.raises(ValueError, match="outside"):
        Sch.shard_jobs([], 8, 8)
