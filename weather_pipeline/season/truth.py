"""Truth sources by name, as listed in a season config's `truths`.

Every source has the same interface, `source(times, variables)` ->
DataArray (time, variable, lat, lon), so wbx_scoring.fetch_targets and the
season scorer treat them identically. Truth is chosen independently of the
initial-condition source: ERA5-initialised forecasts are scored against
ERA5 AND against the IFS analysis, because the paper shows rankings for
extremes depend on that choice (models fine-tuned on IFS analyses lose
skill when verified against ERA5, especially over mountains).
"""

from __future__ import annotations


def build_truth(name: str, **kwargs):
    if name == "era5":
        # ARCO-ERA5 rather than the Arraylake repo: the same ERA5 up to the
        # last final month, then ERA5T, so a recent season is fully covered.
        from ..datasource import ARCOERA5Source
        return ARCOERA5Source()
    if name == "ifs_fc0":
        from ..ifs_opendata import IFSOpenDataFC0Source
        return IFSOpenDataFC0Source(**kwargs)
    raise ValueError(f"Unknown truth '{name}'")
