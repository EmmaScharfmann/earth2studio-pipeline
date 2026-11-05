"""
Event diagnostics: how the forecasts of ONE target time evolved as the
event approached.

For a target valid time (e.g. the peak afternoon of a heatwave), every
initialisation of the season that reaches it is collected, from the longest
lead down to the shortest, at a few sites and as a land-area average. The
result shows when each model first saw the event, how its ensemble
narrowed, and whether the truth fell inside it.

Sites are grid cells (nearest 0.25 deg cell to the given coordinates): a
cell is a ~25 km average, so its value is expected to sit below a station's
record maximum. When the season has a station truth, the observation of the
nearest station within SITE_STATION_KM of each site is added as truth
"stations" (an on-the-hour reading, so a little below the day's maximum).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .config import SeasonConfig
from .forecast import open_forecast
from .schedule import build_jobs, is_done
from .scoring import TruthCache, land_weights

# Météo-France synoptic stations at the centre of the June 2026 records,
# as (lat, lon); used to pick the nearest grid cell.
HEATWAVE_SITES = {
    "Bordeaux": (44.83, -0.69),     # Bordeaux-Mérignac: 42.5 °C on 23 June 2026
    "Nantes": (47.15, -1.61),
    "Rennes": (48.07, -1.73),
    "Paris": (48.82, 2.34),
    "Lyon": (45.73, 5.08),
}


SITE_STATION_KM = 10.0


def _nearest(da: xr.DataArray, lat: float, lon: float) -> xr.DataArray:
    lon360 = lon % 360
    i = int(np.abs(da["lat"].values - lat).argmin())
    dlon = np.abs((da["lon"].values - lon360 + 180) % 360 - 180)
    j = int(dlon.argmin())
    if abs(da["lat"].values[i] - lat) > 0.25 or dlon[j] > 0.25:
        raise ValueError(f"({lat}, {lon}) is outside the region's grid")
    return da.isel(lat=i, lon=j)


def reaching(cfg: SeasonConfig, valid_time, models: list[str] | None = None) -> list:
    """Finished jobs whose forecast reaches `valid_time`, longest lead first."""
    t = pd.Timestamp(valid_time)
    leads = set(cfg.lead_times)
    jobs = [j for j in build_jobs(cfg, models)
            if t - pd.Timestamp(j.init_time) in leads and is_done(j, cfg)]
    return sorted(jobs, key=lambda j: (j.model, j.init_time))


def convergence(cfg: SeasonConfig, valid_time, sites: dict | None = None,
                models: list[str] | None = None, truth_sources: dict | None = None) -> xr.Dataset:
    """Forecasts of one valid time from every init that reaches it.

    Returns a Dataset with
      forecast (model, lead_time, ensemble, site)   ensemble members, K
      observed (truth, site)                         each truth at that time
    where site includes "land mean" (area-weighted over the land domain)
    and lead_time is valid_time - init_time. Leads a model has no finished
    forecast for are NaN.
    """
    t = pd.Timestamp(valid_time)
    sites = dict(HEATWAVE_SITES if sites is None else sites)
    w = land_weights(cfg)

    def at_sites(da: xr.DataArray) -> xr.DataArray:
        vals = [_nearest(da, *sites[s]).drop_vars(["lat", "lon"], errors="ignore") for s in sites]
        ww = w.sel(lat=da["lat"], lon=da["lon"], method="nearest")
        ww = ww.assign_coords(lat=da["lat"].values, lon=da["lon"].values)
        mean = (da * ww).sum(("lat", "lon")) / ww.sum()
        return xr.concat(vals + [mean], dim=xr.DataArray(list(sites) + ["land mean"], dims="site"))

    by_model = {}
    for job in reaching(cfg, t, models):
        fc = open_forecast(cfg, job)
        # members are exchangeable, and member k of one model has nothing to
        # do with member k of another: renumber 0..n-1 so stacking models
        # does not align them by their (subsampled) original numbers
        fc = fc.assign_coords(ensemble=np.arange(fc.sizes["ensemble"]))
        lead = t - pd.Timestamp(job.init_time)
        one = at_sites(fc.sel(lead_time=lead.to_timedelta64()).load())
        by_model.setdefault(job.model, []).append(one.expand_dims(lead_time=[lead.to_timedelta64()]))
    if not by_model:
        raise LookupError(f"No finished forecast reaches {t}")
    fc = xr.concat([xr.concat(v, dim="lead_time").sortby("lead_time")
                    for v in by_model.values()],
                   dim=xr.DataArray(list(by_model), dims="model"), join="outer")

    truths = []
    for name in [n for n in cfg.truths if n not in ("stations", "stations_lapse")]:
        cache = TruthCache(cfg, name, (truth_sources or {}).get(name))
        truths.append(at_sites(cache.fields([t]).isel(time=0)))
    names = [n for n in cfg.truths if n not in ("stations", "stations_lapse")]
    truth = xr.concat(truths, dim=xr.DataArray(names, dims="truth"))
    if "stations" in cfg.truths:
        truth = xr.concat([truth, _station_values(cfg, t, sites, truth["site"].values)], dim="truth")

    return xr.Dataset({"forecast": fc.transpose("model", "lead_time", "ensemble", "site"),
                       "observed": truth.transpose("truth", "site")},
                      attrs={"valid_time": str(t)})


def _station_values(cfg: SeasonConfig, t: pd.Timestamp, sites: dict, site_labels) -> xr.DataArray:
    """The observation at `t` of the station nearest each site (within
    SITE_STATION_KM), NaN where none; NaN for the land mean."""
    from .observations import StationTruth, _haversine_km
    st = StationTruth(cfg)
    used = st.used()
    obs = st.observed([t]).isel(time=0)
    vals = []
    for label in site_labels:
        if label not in sites:
            vals.append(np.nan); continue
        lat, lon = sites[label]
        d = _haversine_km(lat, lon, used["latitude"].values, used["longitude"].values)
        k = int(np.argmin(d))
        vals.append(float(obs.sel(station=used.index[k])) if d[k] <= SITE_STATION_KM else np.nan)
    return xr.DataArray([vals], dims=("truth", "site"), coords={"truth": ["stations"], "site": list(site_labels)})
