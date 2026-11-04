"""Output selection for the Earth2Studio backends (numpy only, no torch)."""

from collections import OrderedDict

import numpy as np
import pytest

from weather_pipeline.regions import REGIONS, output_selection, to_signed_lon

LAT = np.linspace(90, -90, 721)
LON = np.arange(0, 360, 0.25)


def model_coords(variables=("t2m", "u10m", "z500")):
    return OrderedDict(
        batch=np.empty(0), lead_time=np.array([np.timedelta64(6, "h")]),
        variable=np.array(variables), lat=LAT, lon=LON,
    )


def test_nothing_requested_selects_nothing():
    assert output_selection(model_coords()) == {}


def test_variable_subset_keeps_order_asked_for():
    sel = output_selection(model_coords(), variables=["z500", "t2m"])
    assert list(sel["variable"]) == ["z500", "t2m"]
    assert "lat" not in sel and "lon" not in sel


def test_unknown_variable_fails_with_what_is_available():
    with pytest.raises(ValueError, match=r"\['tp'\].*t2m"):
        output_selection(model_coords(), variables=["t2m", "tp"])


def test_france_crop_crosses_the_meridian_west_to_east():
    sel = output_selection(model_coords(), region=REGIONS["france"])
    lon = sel["lon"]
    # values come from the model's own grid, so Earth2Studio's equality
    # matching always succeeds
    assert np.isin(lon, LON).all() and np.isin(sel["lat"], LAT).all()
    signed = to_signed_lon(lon)
    assert signed[0] == -7.0 and signed[-1] == 11.0
    assert np.all(np.diff(signed) > 0)        # west -> east, no gap or repeat
    assert lon.size == (18 / 0.25) + 1
    assert sel["lat"].size == (13 / 0.25) + 1
    assert np.all(np.diff(sel["lat"]) < 0)    # keeps the grid's own order


def test_curvilinear_grid_is_rejected():
    coords = model_coords()
    coords["lat"] = np.zeros((10, 10))
    with pytest.raises(ValueError, match="regular 1-D"):
        output_selection(coords, region=REGIONS["france"])
