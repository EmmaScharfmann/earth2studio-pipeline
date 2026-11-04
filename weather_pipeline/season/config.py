"""
Season configuration: which dates, models, region, lead times and truths.

The period is an input, never a constant. Either name a meteorological
season and year::

    period: {season: JJA, year: 2026}

or give explicit dates (inclusive)::

    period: {start: 2026-06-01, end: 2026-08-31}

DJF follows the meteorological convention: DJF 2026 is December 2025 to
February 2026, i.e. the winter that ENDS in the named year.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Meteorological seasons as (first month, number of months).
SEASONS: dict[str, tuple[int, int]] = {
    "DJF": (12, 3), "MAM": (3, 3), "JJA": (6, 3), "SON": (9, 3),
}

# Natively probabilistic checkpoints only: the study is ensemble-only, and a
# deterministic model re-run N times would give identical "members".
ENSEMBLE_MODELS = {"aifs-ens1.0", "aurora1.5-ens-6h", "atlas-crps"}

# Truth datasets and the period each can serve. Checked up front so a run
# never discovers missing truth after hours of GPU time.
TRUTHS: dict[str, tuple[str | None, str | None]] = {
    # Arraylake ERA5: from 1940, lags real time by a few months
    "era5": ("1940-01-01", None),
    # step-0 HRES fields from ECMWF open data on AWS (the paper's
    # "HRES-fc0"); the bucket starts in 2024
    "ifs_fc0": ("2024-01-01", None),
}

# Cutoffs of each model's training data (last year seen in any phase), from
# the model cards. A season overlapping these is in-sample for that model.
TRAINING_END_YEAR = {
    "aifs-ens1.0": 2023,     # ERA5 1979-2022, then operational analyses
    "aurora1.5-ens-6h": 2023,   # HRES analysis fine-tuning 2018-2023
    "atlas-crps": 2019,      # ERA5 1980-2019
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    members: int

    def __post_init__(self):
        if self.name not in ENSEMBLE_MODELS:
            raise ValueError(
                f"'{self.name}' is not one of the ensemble models "
                f"{sorted(ENSEMBLE_MODELS)}"
            )
        if self.members < 2:
            raise ValueError(f"{self.name}: an ensemble needs >= 2 members")


@dataclass(frozen=True)
class SeasonConfig:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp                    # inclusive, last init date
    models: tuple[ModelSpec, ...]
    init_hours: tuple[int, ...] = (0,)
    init_every_days: int = 1
    lead_every_hours: int = 6
    lead_max_hours: int = 120
    region: str = "france"
    variable: str = "t2m"
    ic_source: str = "era5"
    truths: tuple[str, ...] = ("era5", "ifs_fc0")
    # thresholds for extremes: climatological quantile levels (1.0 = record)
    quantiles: tuple[float, ...] = (0.9, 0.95, 0.975, 0.99, 0.999, 1.0)
    climatology_years: tuple[int, int] = (1979, 2017)
    # every model is scored at this many members (randomly subsampled with a
    # fixed seed), so size-dependent metrics like the record miss rate are
    # comparable; None means the smallest ensemble in the config
    common_members: int | None = None
    output_dir: str = "outputs/season"

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"end {self.end.date()} is before start {self.start.date()}")
        if self.lead_max_hours % self.lead_every_hours:
            raise ValueError("lead_max_hours must be a multiple of lead_every_hours")
        if self.lead_every_hours % 6:
            raise ValueError("lead_every_hours must be a multiple of 6 "
                             "(the native step of the AIFS and Atlas models)")
        for truth in self.truths:
            if truth not in TRUTHS:
                raise ValueError(f"Unknown truth '{truth}'. Known: {sorted(TRUTHS)}")
            first, last = TRUTHS[truth]
            # valid times run up to lead_max_hours past the last init
            last_valid = self.end + pd.Timedelta(hours=max(self.init_hours) + self.lead_max_hours)
            if first and self.start < pd.Timestamp(first):
                raise ValueError(f"Truth '{truth}' starts {first}, before the period")
            if last and last_valid > pd.Timestamp(last):
                raise ValueError(f"Truth '{truth}' ends {last}, before the last valid time")
        if self.common_members is not None:
            small = min(m.members for m in self.models)
            if self.common_members > small:
                raise ValueError(f"common_members={self.common_members} exceeds the "
                                 f"smallest ensemble ({small})")

    # ---- derived quantities -------------------------------------------------

    @property
    def n_members_scored(self) -> int:
        return self.common_members or min(m.members for m in self.models)

    @property
    def init_times(self) -> pd.DatetimeIndex:
        days = pd.date_range(self.start, self.end, freq=f"{self.init_every_days}D")
        return pd.DatetimeIndex(sorted(
            d + pd.Timedelta(hours=h) for d in days for h in self.init_hours
        ))

    @property
    def lead_times(self) -> pd.TimedeltaIndex:
        return pd.to_timedelta(
            range(self.lead_every_hours, self.lead_max_hours + 1, self.lead_every_hours),
            unit="h",
        )

    @property
    def n_steps(self) -> int:
        """Forecast length in 6-hour steps, run_benchmark's --n-steps unit."""
        return self.lead_max_hours // 6

    def in_sample_models(self) -> list[str]:
        """Models whose training data overlaps the period."""
        return [m.name for m in self.models
                if TRAINING_END_YEAR.get(m.name, 0) >= self.start.year]

    def describe(self) -> str:
        n = len(self.init_times)
        lines = [
            f"{self.name}: {n} initialisations, {self.start.date()} to {self.end.date()}, "
            f"hours {list(self.init_hours)}, every {self.init_every_days} day(s)",
            f"  leads {self.lead_every_hours}-{self.lead_max_hours} h, region {self.region}, "
            f"variable {self.variable}, ICs {self.ic_source}, truths {list(self.truths)}",
            "  models: " + ", ".join(f"{m.name} x{m.members}" for m in self.models)
            + f" (scored at {self.n_members_scored} members)",
        ]
        if self.in_sample_models():
            lines.append(f"  WARNING: in-sample for {self.in_sample_models()}")
        return "\n".join(lines)


def resolve_period(period: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """{season, year} or {start, end} -> (first, last) init date."""
    if "season" in period:
        if "start" in period or "end" in period:
            raise ValueError("Give either season+year or start+end, not both")
        season = str(period["season"]).upper()
        if season not in SEASONS:
            raise ValueError(f"Unknown season '{season}'. Known: {sorted(SEASONS)}")
        first_month, n_months = SEASONS[season]
        year = int(period["year"])
        if season == "DJF":
            year -= 1                     # DJF 2026 starts in December 2025
        start = pd.Timestamp(year=year, month=first_month, day=1)
        end = start + pd.DateOffset(months=n_months) - pd.Timedelta(days=1)
        return start, end
    try:
        return pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    except KeyError as exc:
        raise ValueError("period needs either {season, year} or {start, end}") from exc


def from_dict(raw: dict) -> SeasonConfig:
    raw = dict(raw)
    start, end = resolve_period(raw.pop("period"))
    models = tuple(ModelSpec(name, int(spec["members"]))
                   for name, spec in raw.pop("models").items())
    init = raw.pop("init", {}) or {}
    leads = raw.pop("lead_hours", {}) or {}
    thresholds = raw.pop("thresholds", {}) or {}

    kwargs = dict(name=raw.pop("name"), start=start, end=end, models=models)
    if "hours" in init:
        kwargs["init_hours"] = tuple(int(h) for h in init["hours"])
    if "every_days" in init:
        kwargs["init_every_days"] = int(init["every_days"])
    if "every" in leads:
        kwargs["lead_every_hours"] = int(leads["every"])
    if "max" in leads:
        kwargs["lead_max_hours"] = int(leads["max"])
    if "quantiles" in thresholds:
        kwargs["quantiles"] = tuple(float(q) for q in thresholds["quantiles"])
    if "climatology_years" in thresholds:
        kwargs["climatology_years"] = tuple(int(y) for y in thresholds["climatology_years"])
    if "truths" in raw:
        kwargs["truths"] = tuple(raw.pop("truths"))

    known = {f.name for f in dataclasses.fields(SeasonConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    kwargs.update(raw)
    return SeasonConfig(**kwargs)


def load(path: str | Path, **overrides) -> SeasonConfig:
    """Load a YAML config. `overrides` replace top-level keys, e.g.
    load(path, period={"season": "JJA", "year": 2025}) to rerun another summer."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    raw.update(overrides)
    return from_dict(raw)
