from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alerts_forecast.data import load_regions
from alerts_forecast.features import (
    FEATURE_SETS, NEIGHBORS, active_time_before, build_features, count_starts, minutes_since_last_start,
)
from alerts_forecast.leakage import truncate_regions, truncation_violations
from test_leakage import ONE_MIN, probe_times, random_alerts

REAL_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "volunteer_data_en.csv"
TARGET = "Poltavska oblast"


def synthetic_regions() -> dict:
    names = [TARGET, *NEIGHBORS, "Odeska oblast", "Lvivska oblast"]
    return {n: random_alerts(n=250, seed=10 + i) for i, n in enumerate(names)}


def ns(ts):
    return pd.DatetimeIndex(ts).tz_convert("UTC").tz_localize(None).as_unit("ns").to_numpy()


def test_feature_sets_only_name_existing_columns():
    regions = synthetic_regions()
    cols = set(build_features(regions, TARGET, pd.DatetimeIndex([regions[TARGET]["started_at"].iloc[50]])).columns)
    for name, names in FEATURE_SETS.items():
        assert set(names) <= cols, name
        assert len(names) == len(set(names))
    assert "y" not in cols and "active" not in cols


def test_count_starts_matches_a_slow_loop():
    alerts = random_alerts()
    times = probe_times(alerts, n=60)
    w = pd.Timedelta(hours=3)
    got = count_starts(alerts, ns(times), w)
    slow = [((alerts["started_at"] > t - w) & (alerts["started_at"] <= t)).sum() for t in times]
    assert got.tolist() == slow


def test_active_time_matches_a_slow_loop_and_clips_running_alert_at_t():
    alerts = random_alerts()
    times = probe_times(alerts, n=60)
    got = active_time_before(alerts, ns(times))
    for t, g in zip(times, got):
        clipped = np.minimum(alerts["finished_at"], t) - alerts["started_at"]
        slow = (clipped[clipped > pd.Timedelta(0)] / ONE_MIN).sum()
        assert abs(g - slow) < 1e-6


def test_minutes_since_last_start_is_capped_and_zero_at_a_start():
    alerts = random_alerts(n=10)
    first = alerts["started_at"].iloc[0]
    t = ns([first - ONE_MIN, first, first + ONE_MIN * 90])
    assert minutes_since_last_start(alerts, t, cap=1440.0).tolist()[:2] == [1440.0, 0.0]
    assert minutes_since_last_start(alerts, t, cap=60.0)[2] == 60.0


def test_neighbour_flag_is_active_between_start_and_end_only():
    regions = synthetic_regions()
    nb = NEIGHBORS[0]
    a = regions[nb].iloc[10]
    key = f"nbr_active_{nb.split()[0].lower()}"
    idx = pd.DatetimeIndex([a["started_at"] - ONE_MIN, a["started_at"], a["finished_at"] - ONE_MIN, a["finished_at"]])
    got = build_features(regions, TARGET, idx)[key].tolist()
    own = regions[nb]
    expect = [float(((own["started_at"] <= t) & (t < own["finished_at"])).any()) for t in idx]
    assert got == expect
    assert got[1] == 1.0 and got[2] == 1.0  # at the start and just before the end


def test_features_on_empty_history_are_defined():
    regions = {r: a.iloc[0:0] for r, a in synthetic_regions().items()}
    out = build_features(regions, TARGET, pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")]))
    assert not out.isna().any().any()


def test_features_do_not_depend_on_the_future_synthetic():
    regions = synthetic_regions()
    times = probe_times(regions[TARGET], n=60)
    fn = lambda d, idx: build_features(d, TARGET, idx)  # noqa: E731
    assert truncation_violations(fn, regions, times, truncate=truncate_regions) == []


def test_features_do_not_depend_on_the_future_in_other_regions_either():
    # a feature that peeks into neighbours' ends would be caught: break one on purpose
    regions = synthetic_regions()
    times = probe_times(regions[TARGET], n=60)

    def leaky(d, idx):
        out = build_features(d, TARGET, idx)
        nb = d[NEIGHBORS[0]]
        out["nbr_ends_soon"] = [float((nb["finished_at"] - t < pd.Timedelta(minutes=30))[nb["finished_at"] > t].any()) for t in idx]
        return out

    assert len(truncation_violations(leaky, regions, times, truncate=truncate_regions)) > 0


@pytest.mark.skipif(not REAL_FILE.exists(), reason="raw data not downloaded")
def test_features_do_not_depend_on_the_future_real_data():
    regions, data_end = load_regions(REAL_FILE)
    assert "Luhanska oblast" not in regions and TARGET in regions
    recent = regions[TARGET].loc[regions[TARGET]["started_at"] >= data_end - pd.Timedelta(days=60)]
    times = probe_times(recent.reset_index(drop=True), n=80)
    fn = lambda d, idx: build_features(d, TARGET, idx)  # noqa: E731
    assert truncation_violations(fn, regions, times, truncate=truncate_regions) == []
