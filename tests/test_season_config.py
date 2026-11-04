import pandas as pd
import pytest

from weather_pipeline.season import config as C

BASE = {"name": "t", "models": {"aifs-ens1.0": {"members": 10},
                                "atlas-crps": {"members": 16}}}


def make(**kw):
    return C.from_dict({**BASE, **kw})


def test_summer_2026_by_season_name():
    c = make(period={"season": "JJA", "year": 2026})
    assert c.start == pd.Timestamp("2026-06-01") and c.end == pd.Timestamp("2026-08-31")
    assert len(c.init_times) == 92
    assert c.in_sample_models() == []                # out of sample for all


def test_djf_is_the_winter_ending_in_the_year():
    c = make(period={"season": "DJF", "year": 2026})
    assert c.start == pd.Timestamp("2025-12-01") and c.end == pd.Timestamp("2026-02-28")


def test_explicit_dates_and_stride():
    c = make(period={"start": "2026-07-01", "end": "2026-07-10"},
             init={"hours": [0, 12], "every_days": 2})
    assert len(c.init_times) == 10                    # 5 days x 2 hours
    assert c.init_times[1] == pd.Timestamp("2026-07-01T12")


def test_leads_and_steps():
    c = make(period={"season": "JJA", "year": 2026}, lead_hours={"every": 12, "max": 120})
    assert c.lead_times[0] == pd.Timedelta(hours=12) and len(c.lead_times) == 10
    assert c.n_steps == 20


def test_in_sample_years_are_flagged():
    c = make(period={"season": "JJA", "year": 2022}, truths=["era5"])
    assert c.in_sample_models() == ["aifs-ens1.0", "atlas-crps"][:1]
    assert "WARNING" in c.describe()


def test_truth_coverage_checked_up_front():
    with pytest.raises(ValueError, match="ifs_fc0"):
        make(period={"season": "JJA", "year": 2022})   # default truths incl. ifs_fc0


def test_scored_members_default_to_smallest_ensemble():
    assert make(period={"season": "JJA", "year": 2026}).n_members_scored == 10
    with pytest.raises(ValueError, match="smallest"):
        make(period={"season": "JJA", "year": 2026}, common_members=12)


@pytest.mark.parametrize("bad", [
    {"models": {"aifs1.0": {"members": 10}}},          # deterministic
    {"models": {"aifs-ens1.0": {"members": 1}}},
    {"lead_hours": {"every": 3, "max": 120}},
    {"typo_key": 1},
])
def test_invalid_configs_fail(bad):
    with pytest.raises(ValueError):
        make(period={"season": "JJA", "year": 2026}, **bad)


def test_shipped_config_loads():
    c = C.load("configs/summer2026_france.yaml")
    assert c.name == "summer2026_france"
    assert [m.name for m in c.models] == ["aifs-ens1.0", "aurora1.5-ens-6h"]
    other = C.load("configs/summer2026_france.yaml", period={"season": "JJA", "year": 2025})
    assert other.start.year == 2025
