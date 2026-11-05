"""
Station observations as a truth for the season: Météo-France hourly 2 m
temperature, mapped to the forecast grid.

Truth names (in a season config's `truths`):

    stations         forecast at the nearest LAND grid cell within 30 km of the
                     station (Wessel et al. with HadISD); a coastal station
                     whose nearest cell is sea uses the nearest land cell
    stations_lapse   the same cell, moved to the station's height with the
                     standard lapse rate (station_interp.lapse_correct). Helps
                     by day, can hurt in night-time inversions (findings.md):
                     reported next to the primary, not instead of it.

Observations come from weather_pipeline.stations.load_stations (QC: active
over the whole window and at least 90 % complete, a fixed holdout flag for
the downscaler), over every valid time the season can reach, and are kept
at 00/06/12/18 UTC only. Saved once per season under
<season dir>/stations/: observations.nc, qc_report.csv, mapping.csv.

A missing observation drops out of that valid time only: the scorer gives
it weight 0 at that lead (metrics.case_statistics accepts time-varying
weights). Stations are weighted equally.

Heights for the lapse correction are the IFS orography of the cell (the grid
the land mask comes from). The models' own orographies differ somewhat from
it; the correction is a diagnostic of height mismatch, not a model-exact
downscaling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..regions import to_signed_lon
from ..station_interp import STD_LAPSE, lapse_correct
from .config import SeasonConfig

logger = logging.getLogger(__name__)

STATION_TRUTHS = {"stations": False, "stations_lapse": True}    # name -> lapse-corrected?
MAX_DISTANCE_KM = 30.0
SYNOPTIC_HOURS = (0, 6, 12, 18)
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat, dlon = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class StationTruth:
    """Observations, the station -> grid-cell mapping, and both directions
    of moving data between them, for one season."""

    def __init__(self, cfg: SeasonConfig, provider=None, max_km: float = MAX_DISTANCE_KM):
        from .scoring import _root
        self.cfg = cfg
        self.dir = _root(cfg) / "stations"
        self._provider = provider
        self.max_km = max_km
        self._obs: xr.Dataset | None = None
        self._map: pd.DataFrame | None = None

    # ---- observations ---------------------------------------------------------

    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Every valid time the season's forecasts can reach."""
        first = self.cfg.start + pd.Timedelta(hours=min(self.cfg.init_hours))
        last = self.cfg.end + pd.Timedelta(hours=max(self.cfg.init_hours) + self.cfg.lead_max_hours)
        return first, last

    @property
    def provider(self):
        if self._provider is None:
            from ..stations import MeteoFranceHourly
            self._provider = MeteoFranceHourly()
        return self._provider

    def observations(self) -> xr.Dataset:
        """(time, station) at synoptic hours, QC'd. Cached per season."""
        if self._obs is not None:
            return self._obs
        path = self.dir / "observations.nc"
        if path.exists():
            self._obs = xr.open_dataset(path).load()
            return self._obs
        from ..stations import load_stations
        start, end = self.window()
        cache = Path(self.cfg.output_dir) / "stations_cache"
        logger.info("stations: fetching %s .. %s for %s", start, end, self.cfg.region)
        ds, report = load_stations(self.provider, self.cfg.region, start, end,
                                   variables=[self.cfg.variable], cache_dir=str(cache))
        times = pd.DatetimeIndex(ds["time"].values)
        ds = ds.isel(time=np.flatnonzero(np.isin(times.hour, SYNOPTIC_HOURS)))
        self.dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(self.dir / "qc_report.csv")
        out = ds.copy()
        out["name"] = out["name"].astype(str)
        out["station"] = out["station"].astype(str)
        tmp = path.with_suffix(".part")
        out.to_netcdf(tmp)
        tmp.rename(path)
        logger.info("stations: %d kept of %d candidates", ds.sizes["station"], len(report))
        self._obs = xr.open_dataset(path).load()
        return self._obs

    # ---- mapping ----------------------------------------------------------------

    def mapping(self) -> pd.DataFrame:
        """One row per observed station: its land cell, distance, heights,
        and whether it is used (mapped) and why not."""
        if self._map is not None:
            return self._map
        path = self.dir / "mapping.csv"
        if path.exists():
            self._map = pd.read_csv(path, index_col="station", dtype={"station": str})
            return self._map
        from .scoring import elevation, land_weights
        obs = self.observations()
        w = land_weights(self.cfg)
        h = elevation(self.cfg)
        lat2, lon2 = np.meshgrid(w["lat"].values, w["lon"].values, indexing="ij")
        land = (w > 0).values
        clat, clon = lat2[land], lon2[land]
        cheight = h.sel(lat=w["lat"], lon=w["lon"], method="nearest").values[land]
        slat = obs["latitude"].values
        slon = obs["longitude"].values
        d = _haversine_km(slat[:, None], slon[:, None], clat[None, :], to_signed_lon(clon)[None, :])
        k = d.argmin(axis=1)
        dist = d[np.arange(len(slat)), k]
        df = pd.DataFrame({
            "name": obs["name"].values.astype(str),
            "latitude": slat, "longitude": slon,
            "elevation": obs["elevation"].values,
            "holdout": obs["holdout"].values.astype(bool),
            "cell_lat": clat[k], "cell_lon": clon[k],
            "cell_height": cheight[k], "distance_km": dist,
        }, index=pd.Index(obs["station"].values.astype(str), name="station"))
        df["height_diff"] = df["elevation"] - df["cell_height"]
        df["mapped"] = df["distance_km"] <= self.max_km
        df["reason"] = np.where(df["mapped"], "", f"no land cell within {self.max_km:.0f} km")
        df.to_csv(path)
        self._map = df
        logger.info("stations: %d mapped, %d without a land cell within %.0f km",
                    int(df["mapped"].sum()), int((~df["mapped"]).sum()), self.max_km)
        return df

    def used(self) -> pd.DataFrame:
        return self.mapping()[lambda m: m["mapped"]]

    # ---- moving data ------------------------------------------------------------

    def at_stations(self, da: xr.DataArray, lapse: bool = False, gamma: float = STD_LAPSE) -> xr.DataArray:
        """A field on the season grid (..., lat, lon) at the used stations:
        (..., station). With lapse=True, moved to the station heights."""
        m = self.used()
        lat = da["lat"].values
        lon = da["lon"].values
        i = np.abs(lat[None, :] - m["cell_lat"].values[:, None]).argmin(axis=1)
        dlon = np.abs((lon[None, :] - m["cell_lon"].values[:, None] + 180) % 360 - 180)
        j = dlon.argmin(axis=1)
        if (np.abs(lat[i] - m["cell_lat"].values) > 1e-6).any() or \
                (dlon[np.arange(len(j)), j] > 1e-6).any():
            raise ValueError("field is not on the grid the stations were mapped to")
        out = da.isel(lat=xr.DataArray(i, dims="station"), lon=xr.DataArray(j, dims="station"))
        out = out.drop_vars(["lat", "lon"], errors="ignore").assign_coords(station=m.index.values)
        if lapse:
            out = lapse_correct(out, xr.DataArray(m["elevation"].values, dims="station"),
                                xr.DataArray(m["cell_height"].values, dims="station"), gamma)
        return out

    def observed(self, valid_times) -> xr.DataArray:
        """(time, station) observations at the used stations; NaN where missing."""
        obs = self.observations()[self.cfg.variable].sel(station=self.used().index.values)
        t = pd.DatetimeIndex(pd.to_datetime(list(valid_times))).values.astype("datetime64[ns]")
        return obs.reindex(time=t)


def truths_at_stations(cfg: SeasonConfig, truths=("era5", "ifs_fc0"), st: StationTruth | None = None) -> xr.Dataset:
    """Each gridded truth against the station observations, at every valid
    time cached for it (i.e. every valid time the scorer used).

    Which truth is closer to what was measured, and the representativeness
    floor: a truth's error at stations contains no forecast error at all, so
    no 0.25 deg forecast can be expected to beat it there.

    Returns dims (truth, mapping, time) for mean error and RMSE over stations,
    and (truth, mapping, station) for per-station mean error and RMSE;
    mapping is 'nearest_land' or 'lapse' (not called 'method': that is a
    keyword of xarray's .sel). Error = truth minus observation.
    """
    from .scoring import TruthCache, _root
    st = st or StationTruth(cfg)
    per_time, per_station = [], []
    names = [t for t in truths if t in cfg.truths and (_root(cfg) / "truth" / t).exists()]
    if not names:
        raise FileNotFoundError("no gridded truth cached yet; run --stage score first")
    for name in names:
        files = sorted((_root(cfg) / "truth" / name).glob("*.nc"))
        times = [pd.to_datetime(f.stem, format="%Y%m%d%H") for f in files]
        fields = TruthCache(cfg, name).fields(times)                    # all cached: no fetch
        obs = st.observed(times)
        rows_t, rows_s = [], []
        for mapping, lapse in (("nearest_land", False), ("lapse", True)):
            at = st.at_stations(fields, lapse=lapse)
            err = (at - obs).transpose("time", "station")
            ok = err.notnull()
            rows_t.append(xr.Dataset({
                "mean_err": err.mean("station"),
                "rmse": np.sqrt((err ** 2).mean("station")),
                "n": ok.sum("station"),
            }).expand_dims(mapping=[mapping]))
            rows_s.append(xr.Dataset({
                "station_mean_err": err.mean("time"),
                "station_rmse": np.sqrt((err ** 2).mean("time")),
            }).expand_dims(mapping=[mapping]))
        per_time.append(xr.concat(rows_t, dim="mapping").expand_dims(truth=[name]))
        per_station.append(xr.concat(rows_s, dim="mapping").expand_dims(truth=[name]))
    # truths can cover different valid times: outer join, NaN where one lacks a time
    out = xr.merge([xr.concat(per_time, dim="truth", join="outer"),
                    xr.concat(per_station, dim="truth", join="outer")], join="outer")
    m = st.used()
    return out.assign_coords(latitude=("station", m["latitude"].values),
                             longitude=("station", m["longitude"].values),
                             elevation=("station", m["elevation"].values))
