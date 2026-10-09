import numpy as np
import pandas as pd
import pytest

from alerts_forecast.data import load_volunteer, validate_alerts
from alerts_forecast.target import build_frame, main_sample, onset_within, state_at


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-01 {s}", tz="UTC").as_unit("ns")


def make_alerts(rows) -> pd.DataFrame:
    """rows: (start 'HH:MM', end 'HH:MM', naive)"""
    return pd.DataFrame(
        {
            "started_at": [ts(r[0]) for r in rows],
            "finished_at": [ts(r[1]) for r in rows],
            "naive": [r[2] for r in rows],
        }
    )


ALERTS = make_alerts([("10:00", "10:40", False), ("11:00", "11:30", True)])


def at(*hhmm):
    return pd.DatetimeIndex([ts(x) for x in hhmm])


def test_active_boundaries():
    s = state_at(ALERTS, at("09:59", "10:00", "10:39", "10:40"))
    assert s["active"].tolist() == [False, True, True, False]


def test_minutes_since_last_end_and_naive_flag():
    s = state_at(ALERTS, at("09:00", "10:00", "11:00", "12:00"))
    # before any alert: unknown; during an alert: not ended yet; later: counted from the last end
    assert np.isnan(s["minutes_since_last_end"].iloc[0])
    assert np.isnan(s["minutes_since_last_end"].iloc[1])
    assert np.isnan(s["minutes_since_last_end"].iloc[2])
    assert s["minutes_since_last_end"].iloc[3] == 30
    assert s["prev_naive"].tolist() == [False, False, True, True]


def test_label_window_is_open_on_the_left_and_closed_on_the_right():
    one_hour = pd.Timedelta(hours=1)
    y = onset_within(ALERTS, at("08:59", "09:00", "09:59", "10:00", "10:01"), one_hour)
    # 08:59 -> (08:59, 09:59] no start; 09:00 -> (09:00, 10:00] yes, a start at t+H counts;
    # 09:59 yes; 10:00 -> the start at 10:00 is the past, but (10:00, 11:00] has the 11:00 start; 10:01 same
    assert y.tolist() == [False, True, True, True, True]
    y2 = onset_within(ALERTS, at("10:00"), pd.Timedelta(minutes=30))
    assert y2.tolist() == [False]  # start at 10:00 is not in (10:00, 10:30]


def test_empty_table_means_nothing_is_known():
    empty = ALERTS.iloc[0:0]
    s = state_at(empty, at("10:00"))
    assert s["active"].tolist() == [False] and np.isnan(s["minutes_since_last_end"].iloc[0])
    assert onset_within(empty, at("10:00"), pd.Timedelta(hours=1)).tolist() == [False]


def test_label_after_last_alert_is_false():
    assert onset_within(ALERTS, at("12:00"), pd.Timedelta(hours=6)).tolist() == [False]


def test_frame_never_uses_windows_beyond_data_end():
    data_end = ts("13:07")
    frame = build_frame(ALERTS, data_end, horizon_h=3)
    assert frame.index.max() + pd.Timedelta(hours=3) <= data_end
    assert frame.index.max() == ts("10:00")  # floor(13:07 - 3h) to 15 min
    assert frame.index.min() == ts("10:00")  # grid starts at the first alert


def test_main_sample_has_only_inactive_moments():
    frame = build_frame(ALERTS, ts("20:00"), horizon_h=1)
    assert not main_sample(frame)["active"].any()
    assert frame["active"].any()


def test_validate_rejects_overlap_and_negative_duration():
    with pytest.raises(ValueError, match="overlapping"):
        validate_alerts(make_alerts([("10:00", "10:40", False), ("10:30", "11:00", False)]))
    with pytest.raises(ValueError, match="before it starts"):
        validate_alerts(make_alerts([("10:40", "10:00", False)]))


def _write_csv(path, rows):
    header = "region,started_at,finished_at,naive\n"
    path.write_text(header + "".join(",".join(map(str, r)) + "\n" for r in rows))
    return path


def test_loader_end_of_data_comes_from_the_whole_file(tmp_path):
    p = _write_csv(
        tmp_path / "v.csv",
        [
            ("Poltavska oblast", "2026-01-01 10:00:00+00:00", "2026-01-01 10:40:00+00:00", False),
            ("Odeska oblast", "2026-01-01 15:00:00+00:00", "2026-01-01 15:20:00+00:00", False),
        ],
    )
    alerts, data_end = load_volunteer(p)
    assert len(alerts) == 1
    assert data_end == pd.Timestamp("2026-01-01 15:20:00", tz="UTC")


def test_loader_rejects_non_utc_and_unknown_region(tmp_path):
    p = _write_csv(
        tmp_path / "v.csv",
        [("Poltavska oblast", "2026-01-01 12:00:00+02:00", "2026-01-01 12:40:00+02:00", False)],
    )
    with pytest.raises(ValueError, match="UTC"):
        load_volunteer(p)
    p2 = _write_csv(
        tmp_path / "w.csv",
        [("Odeska oblast", "2026-01-01 10:00:00+00:00", "2026-01-01 10:40:00+00:00", False)],
    )
    with pytest.raises(ValueError, match="no alerts"):
        load_volunteer(p2)
