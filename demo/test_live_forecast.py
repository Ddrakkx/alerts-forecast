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


def test_last_successful_poll_ignores_errors_and_is_not_the_last_event(tmp_path):
    p = tmp_path / "polls.csv"
    p.write_text("polled_at,status\n2026-10-10T12:51:29+00:00,200\n2026-10-10T12:52:29+00:00,304\n"
                 "2026-10-10T12:53:29+00:00,error: OSError\n", encoding="utf-8")
    assert LF.last_successful_poll(p) == pd.Timestamp("2026-10-10 12:52:29", tz="UTC")


def test_gaps_in_the_log_mark_the_history_incomplete(tmp_path):
    p = tmp_path / "polls.csv"
    p.write_text("polled_at,status\n2026-10-10T14:47:29+00:00,200\n2026-10-10T14:48:29+00:00,200\n"
                 "2026-10-10T16:45:00+00:00,200\n2026-10-10T16:46:00+00:00,304\n", encoding="utf-8")
    gaps = LF.log_gaps(p)
    assert gaps[0] == (LF.SNAPSHOT_END, LF.LOGGER_START)
    assert gaps[1] == (pd.Timestamp("2026-10-10 14:48:29", tz="UTC"), pd.Timestamp("2026-10-10 16:45", tz="UTC"))
    st = LF.history_status(pd.Timestamp("2026-10-12 12:00", tz="UTC"), gaps)  # 24 h after the switch-off gap: complete again
    assert st["complete_24h"] is True and st["complete_7d"] is False and len(st["gaps_last_7d"]) == 2
    assert LF.history_status(pd.Timestamp("2026-10-10 17:00", tz="UTC"), gaps)["complete_24h"] is False
