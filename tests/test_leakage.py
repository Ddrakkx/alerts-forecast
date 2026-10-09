from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alerts_forecast.data import load_volunteer
from alerts_forecast.leakage import truncate_at, truncation_violations
from alerts_forecast.target import onset_within, state_at

REAL_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "volunteer_data_en.csv"
ONE_MIN = pd.Timedelta(minutes=1)


def random_alerts(n=300, seed=0) -> pd.DataFrame:
    """Non-overlapping synthetic alerts with gaps from 0 to a few hours."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2026-01-01", tz="UTC").as_unit("ns")
    rows = []
    for _ in range(n):
        t += pd.Timedelta(minutes=int(rng.integers(0, 300)))
        end = t + pd.Timedelta(minutes=int(rng.integers(1, 120)), seconds=int(rng.integers(0, 60)))
        rows.append((t, end, bool(rng.random() < 0.05)))
        t = end
    return pd.DataFrame(rows, columns=["started_at", "finished_at", "naive"])


def probe_times(alerts, n=200, seed=1) -> list:
    """Random moments plus the nasty ones: exactly at starts and ends, one minute around them."""
    rng = np.random.default_rng(seed)
    lo, hi = alerts["started_at"].iloc[0], alerts["finished_at"].iloc[-1]
    span = (hi - lo) / ONE_MIN
    random_t = [lo + pd.Timedelta(minutes=float(rng.random() * span)) for _ in range(n)]
    edges = list(alerts["started_at"].iloc[:40]) + list(alerts["finished_at"].iloc[:40])
    edges += [e + d for e in edges[:20] for d in (-ONE_MIN, ONE_MIN)]
    return random_t + edges


def leaky_state(alerts, times) -> pd.DataFrame:
    """Deliberately wrong feature: minutes until the active alert ends. Uses the future."""
    out = state_at(alerts, times)
    ends = alerts["finished_at"].reset_index(drop=True)
    remaining = []
    for t, active in zip(times, out["active"]):
        started = alerts.index[alerts["started_at"] <= t]
        remaining.append((ends[started[-1]] - t) / ONE_MIN if active and len(started) else np.nan)
    out["remaining_minutes"] = remaining
    return out


def test_state_does_not_depend_on_the_future_synthetic():
    alerts = random_alerts()
    assert truncation_violations(state_at, alerts, probe_times(alerts)) == []


def test_the_check_itself_catches_a_leaky_feature():
    alerts = random_alerts()
    bad = truncation_violations(leaky_state, alerts, probe_times(alerts))
    assert len(bad) > 0


def test_label_depends_only_on_starts_inside_the_window():
    alerts = random_alerts()
    horizon = pd.Timedelta(hours=3)
    for t in probe_times(alerts)[:150]:
        idx = pd.DatetimeIndex([t])
        y = onset_within(alerts, idx, horizon)[0]
        # nothing known at t (starts <= t) may change the label
        future_only = alerts.loc[alerts["started_at"] > t]
        if len(future_only):
            assert onset_within(future_only, idx, horizon)[0] == y
        # nothing after the window may change the label
        window_only = alerts.loc[alerts["started_at"] <= t + horizon]
        assert onset_within(window_only, idx, horizon)[0] == y
        # ends of alerts do not matter for the label at all
        moved = alerts.assign(finished_at=alerts["finished_at"] + pd.Timedelta(hours=2))
        assert onset_within(moved, idx, horizon)[0] == y


def test_truncate_hides_ends_after_t_and_drops_later_alerts():
    alerts = random_alerts(n=20)
    t = alerts["started_at"].iloc[5] + ONE_MIN
    cut = truncate_at(alerts, t)
    assert (cut["started_at"] <= t).all()
    assert len(cut) == 6
    assert cut["finished_at"].iloc[-1] > t  # unknown end is far in the future


@pytest.mark.skipif(not REAL_FILE.exists(), reason="raw data not downloaded")
def test_state_does_not_depend_on_the_future_real_data():
    alerts, data_end = load_volunteer(REAL_FILE)
    recent = alerts.loc[alerts["started_at"] >= data_end - pd.Timedelta(days=60)].reset_index(drop=True)
    assert len(recent) > 50
    assert truncation_violations(state_at, alerts, probe_times(recent, n=150)) == []
