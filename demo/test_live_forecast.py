"""Tests for demo/live_forecast.py on a synthetic eMap log, no network.  Run: python -m pytest demo/"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_forecast as LF  # noqa: E402

P = "Полтавська область"


def snap(raions: dict) -> dict:
    return {"version": 10, "states": {P: {"enabled": False, "type:": "state", "enabled_at": None, "disabled_at": None, "alert_level": None,
            "districts": {r: {"enabled": on, "type": "district", "enabled_at": at, "disabled_at": None, "alert_level": lvl}
                          for r, (on, at, lvl) in raions.items()}}}}


def write_log(d: Path, snapshots: dict, events: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, s in snapshots.items():
        (d / f"snapshot_{name}.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    (d / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")


def test_intervals_union_and_all_clear(tmp_path):
    write_log(tmp_path, {"2026-10-10T120000+0000": snap({"A район": (True, "2026-10-10T11:50:00.000Z", "yellow"), "B район": (False, None, None)})}, [
        {"logged_at": "2026-10-10T12:05:00+00:00", "oblast": P, "raion": "B район", "event": "start", "started_at": "2026-10-10T12:04:30.000Z",
         "alert_level": "red", "seen_between": ["x", "y"]},
        {"logged_at": "2026-10-10T12:10:00+00:00", "oblast": P, "raion": "A район", "event": "end", "started_at": "2026-10-10T11:50:00.000Z",
         "alert_level": "yellow", "ended_between": ["2026-10-10T12:09:00+00:00", "2026-10-10T12:10:00+00:00"]},
        {"logged_at": "2026-10-10T12:20:00+00:00", "oblast": P, "raion": "B район", "event": "end", "started_at": "2026-10-10T12:04:30.000Z",
         "alert_level": "red", "ended_between": ["2026-10-10T12:19:00+00:00", "2026-10-10T12:20:00+00:00"]},
    ])
    closed, open_ = LF.intervals(LF.load_log(tmp_path))
    assert len(closed) == 2 and open_ == {}
    ep = LF.oblast_episodes(closed, open_, P)
    assert len(ep) == 1  # A (11:50-12:10) and B (12:04:30-12:20) overlap: one oblast episode
    assert ep["started_at"].iloc[0] == pd.Timestamp("2026-10-10 11:50", tz="UTC")
    assert LF.poltava_all_clears(closed, open_) == [pd.Timestamp("2026-10-10 12:20", tz="UTC")]  # the poll that saw the last raion off


def test_running_alert_has_unknown_end_and_no_all_clear(tmp_path):
    write_log(tmp_path, {"2026-10-10T120000+0000": snap({"A район": (True, "2026-10-10T11:50:00.000Z", "red")})}, [])
    closed, open_ = LF.intervals(LF.load_log(tmp_path))
    ep = LF.oblast_episodes(closed, open_, P)
    assert ep["finished_at"].iloc[0] == LF.UNKNOWN_END and LF.poltava_all_clears(closed, open_) == []


def test_a_logger_restart_does_not_lose_an_end(tmp_path):
    # alert running at the first snapshot, ended while the logger was down, the second snapshot shows it off
    write_log(tmp_path, {"2026-10-10T120000+0000": snap({"A район": (True, "2026-10-10T11:50:00.000Z", "red")}),
                         "2026-10-10T123000+0000": snap({"A район": (False, None, None)})}, [])
    closed, open_ = LF.intervals(LF.load_log(tmp_path))
    assert open_ == {} and closed[0][3] == pd.Timestamp("2026-10-10 12:30", tz="UTC")  # end known only at the second snapshot
    assert LF.poltava_all_clears(closed, open_) == [pd.Timestamp("2026-10-10 12:30", tz="UTC")]
