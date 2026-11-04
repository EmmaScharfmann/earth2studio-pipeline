import json

import pandas as pd
import pytest

from weather_pipeline import ifs_opendata as I


def test_file_url_layout():
    assert I.file_url("2026-07-15T06").endswith(
        "/20260715/06z/ifs/0p25/oper/20260715060000-0h-oper-fc.grib2")


def test_non_synoptic_hour_is_rejected():
    with pytest.raises(I.TruthUnavailable):
        I.file_url("2026-07-15T03")


def test_find_message_picks_step0_surface_2t():
    lines = [
        {"param": "2t", "levtype": "sfc", "step": "6", "_offset": 1, "_length": 1},
        {"param": "t", "levtype": "pl", "step": "0", "_offset": 2, "_length": 2},
        {"param": "2t", "levtype": "sfc", "step": "0", "_offset": 300, "_length": 40},
    ]
    text = "\n".join(json.dumps(x) for x in lines)
    assert I.find_message(text, "2t") == (300, 40)
    with pytest.raises(I.TruthUnavailable):
        I.find_message(text, "msl")


def test_dates_before_the_archive_fail_without_network(tmp_path):
    src = I.IFSOpenDataFC0Source(cache_dir=tmp_path)
    with pytest.raises(I.TruthUnavailable, match="starts"):
        src.fetch_field(pd.Timestamp("2022-07-01"), "t2m")
