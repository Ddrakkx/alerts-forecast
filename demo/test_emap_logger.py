"""Tests for demo/emap_logger.py, no network.  Run: python -m pytest demo/"""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emap_logger as L  # noqa: E402


def feed(districts: dict, oblast_on=False, oblast_level=None) -> dict:
    return {"version": 10, "states": {"Полтавська область": {
        "enabled": oblast_on, "type:": "state", "enabled_at": None, "disabled_at": None, "alert_level": oblast_level,
        "districts": {name: {"enabled": on, "type": "district", "enabled_at": at, "disabled_at": None, "alert_level": lvl}
                      for name, (on, at, lvl) in districts.items()}}}}


def test_flatten_has_oblast_and_raion_keys():
    st = L.flatten(feed({"Полтавський район": (True, "2026-10-10T12:00:00.000Z", "yellow")}))
    assert st[("Полтавська область", "")]["enabled"] is False
    assert st[("Полтавська область", "Полтавський район")] == {"enabled": True, "enabled_at": "2026-10-10T12:00:00.000Z", "alert_level": "yellow"}


def test_diff_start_level_change_end_and_restart():
    off = L.flatten(feed({"A район": (False, None, None)}))
    yellow = L.flatten(feed({"A район": (True, "T1", "yellow")}))
    red = L.flatten(feed({"A район": (True, "T1", "red")}))
    again = L.flatten(feed({"A район": (True, "T2", "red")}))
    ev = L.diff(off, yellow, "p0", "p1")
    assert [e["event"] for e in ev] == ["start"] and ev[0]["started_at"] == "T1" and ev[0]["seen_between"] == ["p0", "p1"]
    ev = L.diff(yellow, red, "p1", "p2")
    assert ev == [{"oblast": "Полтавська область", "raion": "A район", "event": "level", "started_at": "T1",
                   "from": "yellow", "to": "red", "changed_between": ["p1", "p2"]}]
    ev = L.diff(red, again, "p2", "p3")  # ended and started again between two polls
    assert [e["event"] for e in ev] == ["end", "start"]
    ev = L.diff(again, off, "p3", "p4")
    assert ev[0]["event"] == "end" and ev[0]["ended_between"] == ["p3", "p4"] and ev[0]["alert_level"] == "red"
    assert L.diff(off, off, "p4", "p5") == []


def test_interval_below_one_minute_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="minimum"):
        L.run(interval=30, max_polls=1, out=tmp_path)


class FakeFetcher:
    def __init__(self, responses):
        self.responses = list(responses)

    def get(self):
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_run_writes_snapshot_events_polls_and_waits_at_least_a_minute(tmp_path):
    f0 = feed({"A район": (False, None, None)})
    f1 = feed({"A район": (True, "T1", "red")})
    sleeps = []
    fetch = FakeFetcher([(200, f0, "lm0"), (304, None, "lm0"), (200, f1, "lm1"), OSError("network down"), (200, f1, "lm1")])
    L.run(interval=60, max_polls=5, out=tmp_path, fetcher=fetch, sleep=sleeps.append, clock=lambda: 0.0)
    assert len(list(tmp_path.glob("snapshot_*.json"))) == 1
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["start"] and events[0]["raion"] == "A район"
    rows = (tmp_path / "polls.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 6 and rows[2].split(",")[1] == "304" and "error: OSError" in rows[4]
    assert sleeps == [60, 60, 60, 120]  # never below a minute, longer after an error


def test_fetcher_sends_conditional_headers_and_understands_304():
    seen = []

    class Resp(io.BytesIO):
        status = 200
        headers = {"ETag": '"abc"', "Last-Modified": "Sat, 10 Oct 2026 12:17:41 GMT"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        seen.append(dict(req.header_items()))
        if len(seen) == 1:
            return Resp(json.dumps({"states": {}}).encode())
        raise urllib.error.HTTPError(req.full_url, 304, "Not Modified", {}, None)

    f = L.Fetcher(opener)
    assert f.get()[0] == 200
    assert f.get() == (304, None, "Sat, 10 Oct 2026 12:17:41 GMT")
    assert seen[1].get("If-none-match") == '"abc"' and "If-modified-since" in seen[1]
    assert "alerts-forecast" in seen[0]["User-agent"]


def test_the_minute_holds_across_restarts(tmp_path):
    from datetime import datetime, timezone

    polls = tmp_path / "polls.csv"
    assert L.seconds_to_wait(polls, 60, datetime(2026, 10, 10, 12, 28, tzinfo=timezone.utc)) == 0.0
    polls.write_text("polled_at,status\n2026-10-10T12:27:09+00:00,200\n", encoding="utf-8")
    t = datetime(2026, 10, 10, 12, 27, 29, tzinfo=timezone.utc)
    assert L.seconds_to_wait(polls, 60, t) == 40.0  # a restart 20 s after the last poll waits 40 s
    sleeps = []
    fetch = FakeFetcher([(304, None, "lm")])
    L.run(interval=60, max_polls=1, out=tmp_path, fetcher=fetch, sleep=sleeps.append, clock=lambda: 0.0, now=lambda: t)
    assert sleeps[0] == 40.0
