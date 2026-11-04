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


# ---- mirrors and throttling ----------------------------------------------------

import urllib.error  # noqa: E402

INDEX = json.dumps({"param": "2t", "levtype": "sfc", "step": "0",
                    "_offset": 10, "_length": 4})


class FakeNet:
    """Stands in for urlopen. behaviour[mirror] is a list of outcomes per
    request: an int HTTP status (error) or "ok"; the last one repeats."""

    def __init__(self, monkeypatch, behaviour):
        self.behaviour, self.calls, self.slept = behaviour, [], []
        monkeypatch.setattr(I.urllib.request, "urlopen", self.urlopen)
        monkeypatch.setattr(I, "_sleep", self.slept.append)

    def urlopen(self, request, timeout=None):
        url = request.full_url
        base = next(b for b in self.behaviour if url.startswith(b))
        outcomes = self.behaviour[base]
        n = sum(1 for c in self.calls if c.startswith(base))
        self.calls.append(url)
        outcome = outcomes[min(n, len(outcomes) - 1)]
        if outcome != "ok":
            raise urllib.error.HTTPError(url, outcome, "x", {}, None)
        body = INDEX.encode() if url.endswith(".index") else b"GRIB"
        return _Response(body)


class _Response:
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self.body


A, B = "https://mirror-a", "https://mirror-b"
T = pd.Timestamp("2026-07-15T12")


def test_throttled_mirror_falls_through_to_the_next(monkeypatch):
    net = FakeNet(monkeypatch, {A: [503], B: ["ok"]})
    src = I.IFSOpenDataFC0Source(mirrors=(A, B))
    assert src._download(T, "2t", "sfc") == b"GRIB"
    assert sum(c.startswith(A) for c in net.calls) == I.RETRIES   # A retried fully
    assert len(net.slept) == I.RETRIES - 1                        # with backoff
    assert all(0 <= d <= I.MAX_BACKOFF_S for d in net.slept)
    # index and GRIB bytes came from the same mirror
    assert [c for c in net.calls if c.startswith(B)][0].endswith(".index")


def test_transient_throttling_recovers_on_the_same_mirror(monkeypatch):
    net = FakeNet(monkeypatch, {A: [503, 503, "ok"], B: ["ok"]})
    src = I.IFSOpenDataFC0Source(mirrors=(A, B))
    assert src._download(T, "2t", "sfc") == b"GRIB"
    assert not any(c.startswith(B) for c in net.calls)


def test_missing_file_skips_retries(monkeypatch):
    net = FakeNet(monkeypatch, {A: [404], B: ["ok"]})
    I.IFSOpenDataFC0Source(mirrors=(A, B))._download(T, "2t", "sfc")
    assert sum(c.startswith(A) for c in net.calls) == 1 and net.slept == []


def test_all_mirrors_throttled_is_a_clear_error(monkeypatch):
    FakeNet(monkeypatch, {A: [503], B: [503]})
    with pytest.raises(I.MirrorUnavailable, match="every mirror failed"):
        I.IFSOpenDataFC0Source(mirrors=(A, B))._download(T, "2t", "sfc")


def test_absent_everywhere_is_truth_unavailable(monkeypatch):
    FakeNet(monkeypatch, {A: [404], B: [404]})
    with pytest.raises(I.TruthUnavailable, match="on no mirror"):
        I.IFSOpenDataFC0Source(mirrors=(A, B))._download(T, "2t", "sfc")
