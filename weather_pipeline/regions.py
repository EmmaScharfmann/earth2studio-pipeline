"""
Region handling shared by the backends and (later) the downscaling code.

Deliberately numpy-only: the model environments import this, and they carry
neither weatherbenchX nor earth2studio.

Conventions (match the rest of the package):
  * data longitudes are stored 0..360, latitudes descending (90 .. -90)
  * people write regions with longitudes in -180..180, e.g. France is
    (40, 53, -7, 11)
  * a region that crosses the prime meridian is NOT contiguous in 0..360
    storage (-7..11 is 353..360 plus 0..11). `crop_indices` returns the
    longitude indices ordered west -> east, so the cropped longitude
    coordinate reads 353.0 .. 359.75, 0.0 .. 11.0. It is therefore
    NON-MONOTONIC in 0..360 terms. That keeps it compatible with the
    truth-fetching code (which works in 0..360), but anything that
    interpolates in longitude must first convert with `to_signed_lon`.
"""

from __future__ import annotations

import numpy as np

Region = tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max

REGIONS: dict[str, Region] = {
    "global": (-90, 90, -180, 180),
    "europe": (35, 72, -25, 45),
    "north-atlantic": (20, 70, -80, 10),
    "conus": (24, 50, -125, -66),
    # Metropolitan France plus a margin, so interpolation to stations near
    # the border and coast never reaches the edge of the array.
    "france": (40.0, 53.0, -7.0, 11.0),
}


def parse_region(spec) -> Region:
    """A name from REGIONS, 'lat_min,lat_max,lon_min,lon_max', or a 4-sequence."""
    if isinstance(spec, str):
        if spec in REGIONS:
            return REGIONS[spec]
        try:
            lat0, lat1, lon0, lon1 = (float(x) for x in spec.split(","))
        except ValueError as exc:
            raise ValueError(
                f"Region must be one of {sorted(REGIONS)} or four comma-separated "
                f"numbers 'lat_min,lat_max,lon_min,lon_max'; got {spec!r}"
            ) from exc
        return lat0, lat1, lon0, lon1
    lat0, lat1, lon0, lon1 = (float(x) for x in spec)
    return lat0, lat1, lon0, lon1


def crop_indices(region: Region, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays selecting `region` from a regular lat/lon grid.

    Returns (lat_idx, lon_idx). Latitude keeps the grid's own order. Longitude
    is ordered west -> east even across the prime meridian (see module
    docstring)."""
    lat0, lat1, lon0, lon1 = region
    lat = np.asarray(lat)
    lat_idx = np.flatnonzero((lat >= min(lat0, lat1)) & (lat <= max(lat0, lat1)))

    lon360 = np.asarray(lon) % 360
    a, b = lon0 % 360, lon1 % 360
    if np.isclose(a, b):  # full circle
        lon_idx = np.arange(lon360.size)
    elif a <= b:
        lon_idx = np.flatnonzero((lon360 >= a) & (lon360 <= b))
        lon_idx = lon_idx[np.argsort(lon360[lon_idx], kind="stable")]
    else:  # wraps through 0: west part first, then east part
        west = np.flatnonzero(lon360 >= a)
        east = np.flatnonzero(lon360 <= b)
        lon_idx = np.concatenate([
            west[np.argsort(lon360[west], kind="stable")],
            east[np.argsort(lon360[east], kind="stable")],
        ])

    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(f"Region {region} selects no grid points")
    return lat_idx, lon_idx


def to_signed_lon(lon):
    """0..360 -> -180..180. Use before interpolating across the meridian."""
    lon = np.asarray(lon, dtype="f8")
    return ((lon + 180.0) % 360.0) - 180.0

def output_selection(model_coords, variables=None, region: Region | None = None):
    """Earth2Studio `output_coords` that keep only `variables` inside `region`.

    `model_coords` is the model's own output coordinate system
    (``model.output_coords(model.input_coords())``). The selection is built
    from those exact coordinate values, because Earth2Studio's `map_coords`
    matches lat/lon by equality: values taken from the model itself always
    match, values computed independently might not.

    Returns a plain dict (the caller wraps it in an OrderedDict); an empty
    dict means "write everything", which is Earth2Studio's own default.

    Non-monotonic longitudes for regions crossing 0 deg (see module
    docstring) are fine here: `map_coords` falls back to an index select that
    keeps the requested order.
    """
    selection: dict = {}

    if variables:
        have = [str(v) for v in np.asarray(model_coords["variable"])]
        missing = [v for v in variables if v not in have]
        if missing:
            raise ValueError(
                f"Model does not output {missing}. It outputs: {have}"
            )
        selection["variable"] = np.asarray(list(variables))

    if region is not None:
        lat = np.asarray(model_coords["lat"])
        lon = np.asarray(model_coords["lon"])
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError(
                "Region cropping needs a regular 1-D lat/lon grid; this model "
                f"outputs lat{lat.shape} / lon{lon.shape}."
            )
        lat_idx, lon_idx = crop_indices(region, lat, lon)
        selection["lat"] = lat[lat_idx]
        selection["lon"] = lon[lon_idx]

    return selection
