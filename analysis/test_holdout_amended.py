"""Tests for analysis/holdout_amended.py on synthetic data.  Run: python -m pytest analysis/"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout_amended as HA  # noqa: E402

from alerts_forecast.data import load_regions  # noqa: E402
from alerts_forecast.leakage import UNKNOWN_END  # noqa: E402

OLD_END = pd.Timestamp("2026-10-01 04:50", tz="UTC")


def rows(*recs):
    return pd.DataFrame(recs, columns=["region", "started_at", "finished_at", "naive"])


def test_comparison_separates_running_alerts_from_late_additions_and_ignores_time_format():
    old = rows(("A", "2026-10-01 01:00:00+00:00", "2026-10-01 01:30:00+00:00", False),
               ("B", "2026-10-01 02:00:00+00:00", "2026-10-01 02:20:00+00:00", False))
    new = rows(("A", "2026-10-01T01:00:00Z", "2026-10-01T01:30:00Z", False),              # same record, other time format
               ("B", "2026-10-01 02:00:00+00:00", "2026-10-01 02:45:00+00:00", False),    # changed end
               ("C", "2026-10-01 04:10:00+00:00", "2026-10-01 05:24:00+00:00", False),    # running at the old end: not late
               ("D", "2026-09-20 10:00:00+00:00", "2026-09-20 10:30:00+00:00", True),     # finished long ago, absent before: late
               ("E", "2026-10-01 06:00:00+00:00", "2026-10-01 06:30:00+00:00", False))    # after the old end: not compared
    c = HA.compare_snapshots(old, new, OLD_END)
    assert list(c["running"]["region"]) == ["C"] and list(c["late"]["region"]) == ["D"]
    assert c["removed"].empty and list(c["changed"]["region"]) == ["B"]


def test_overlap_is_merged_with_a_warning_and_clean_files_load_like_the_frozen_loader(tmp_path):
    clean = rows(("Kyiv City", "2026-10-01 01:00:00+00:00", "2026-10-01 01:30:00+00:00", False),
                 ("Kyiv City", "2026-10-01 02:00:00+00:00", "2026-10-01 02:20:00+00:00", True),
                 ("Luhanska oblast", "2022-04-04 19:45:00+00:00", "2026-10-01 03:00:00+00:00", False))
    p = tmp_path / "clean.csv"
    clean.to_csv(p, index=False)
    mine, end, warn = HA.load_robust(p)
    frozen, end2 = load_regions(p)
    assert warn == [] and end == end2 and mine.keys() == frozen.keys() and mine["Kyiv City"].equals(frozen["Kyiv City"])
    bad = pd.concat([clean, rows(("Kyiv City", "2026-10-01 01:20:00+00:00", "2026-10-01 01:40:00+00:00", True))])
    q = tmp_path / "bad.csv"
    bad.to_csv(q, index=False)
    mine, _, warn = HA.load_robust(q)
    k = mine["Kyiv City"]
    assert len(warn) == 1 and len(k) == 2
    assert k["finished_at"].iloc[0] == pd.Timestamp("2026-10-01 01:40", tz="UTC") and bool(k["naive"].iloc[0])


def test_cut_drops_alerts_after_e_and_hides_ends_after_e():
    end = pd.Timestamp("2026-10-12 05:00", tz="UTC")
    a = pd.DataFrame({"started_at": pd.to_datetime(["2026-10-11 20:00", "2026-10-11 22:30", "2026-10-12 02:00"], utc=True),
                      "finished_at": pd.to_datetime(["2026-10-11 20:30", "2026-10-11 23:30", "2026-10-12 02:30"], utc=True),
                      "naive": [False, False, False]})
    cut, e = HA.cut({"X": a}, end)
    assert e == pd.Timestamp("2026-10-11 23:00", tz="UTC")
    assert len(cut["X"]) == 2 and cut["X"]["finished_at"].iloc[1] == UNKNOWN_END
    assert HA.own_starts(a, pd.Timestamp("2026-10-11 19:00", tz="UTC"), e) == 2
