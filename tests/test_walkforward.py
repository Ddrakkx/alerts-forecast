import pytest
import numpy as np
import pandas as pd

from alerts_forecast.baselines import BASELINES, hour_of_week_rate, kyiv_hour_of_week, recent_activity_rate
from alerts_forecast.metrics import brier, day_block_bootstrap, pr_auc, reliability
from alerts_forecast.walkforward import fold_edges, split, walk_forward


def make_sample(n_days=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_days * 96, freq="15min", tz="UTC")
    since = rng.uniform(0, 600, len(idx))
    return pd.DataFrame(
        {"y": rng.random(len(idx)) < 0.4, "minutes_since_last_end": since}, index=idx
    ).rename_axis("t")


def test_folds_cover_the_range_without_gaps_or_overlap():
    a, b = pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-03-20", tz="UTC")
    edges = list(fold_edges(a, b, 7))
    assert edges[0][0] == a and edges[-1][1] == b
    assert all(e1[1] == e2[0] for e1, e2 in zip(edges, edges[1:]))


def test_train_label_windows_end_before_the_test_block():
    sample = make_sample()
    for h in (1, 3, 6):
        fs = pd.Timestamp("2026-03-01", tz="UTC")
        train, test = split(sample, fs, fs + pd.Timedelta(days=7), h)
        assert train.index.max() + pd.Timedelta(hours=h) <= fs
        assert test.index.min() >= fs and test.index.max() < fs + pd.Timedelta(days=7)
        assert not train.index.intersection(test.index).size


def test_rolling_window_keeps_only_recent_training_rows():
    sample = make_sample()
    fs = pd.Timestamp("2026-04-01", tz="UTC")
    train, _ = split(sample, fs, fs + pd.Timedelta(days=7), 3, window_days=30)
    assert train.index.min() > fs - pd.Timedelta(hours=3) - pd.Timedelta(days=30)


def test_fold_predictions_depend_only_on_labels_up_to_start_minus_horizon():
    sample = make_sample()
    h = 3
    for day in ("2026-02-01", "2026-03-01", "2026-04-01"):
        fs = pd.Timestamp(day, tz="UTC")
        fe = fs + pd.Timedelta(days=7)
        a = walk_forward(sample, BASELINES, h, fs, fe)
        flipped = sample.copy()
        late = flipped.index > fs - pd.Timedelta(hours=h)  # the test block and everything after it
        flipped.loc[late, "y"] = ~flipped.loc[late, "y"]
        b = walk_forward(flipped, BASELINES, h, fs, fe)
        for name in BASELINES:
            assert np.allclose(a[name], b[name]), name
        assert (a["y"] != b["y"]).all()  # the test labels themselves did change


def test_later_folds_may_use_earlier_test_labels():
    # sanity check of the test above: the past of fold 2 includes fold 1, so labels there do matter
    sample = make_sample()
    start, end = pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-03-15", tz="UTC")
    a = walk_forward(sample, BASELINES, 3, start, end)
    flipped = sample.copy()
    first = (flipped.index >= start) & (flipped.index < start + pd.Timedelta(days=7))
    flipped.loc[first, "y"] = ~flipped.loc[first, "y"]
    b = walk_forward(flipped, BASELINES, 3, start, end)
    second = a.index >= start + pd.Timedelta(days=7)
    assert not np.allclose(a.loc[second, "constant"], b.loc[second, "constant"])


def test_hour_of_week_follows_kyiv_wall_clock_across_dst():
    # 2026-03-29 is the switch to summer time in Ukraine (UTC+2 -> UTC+3)
    winter = pd.DatetimeIndex([pd.Timestamp("2026-03-28 10:00", tz="UTC")])  # 12:00 Kyiv, Saturday
    summer = pd.DatetimeIndex([pd.Timestamp("2026-04-04 09:00", tz="UTC")])  # 12:00 Kyiv, Saturday
    assert kyiv_hour_of_week(winter)[0] == kyiv_hour_of_week(summer)[0] == 5 * 24 + 12


def test_hour_of_week_baseline_uses_same_slot_mean():
    # January: no summer-time switch. Monday 10:00 UTC rows are [1, 1, 0, 1]; another slot holds zeros,
    # so the global mean (0.375) differs from the slot mean (0.75) and a wrong slot would be noticed.
    mon = [pd.Timestamp("2026-01-05 10:00", tz="UTC") + pd.Timedelta(days=7 * k) for k in range(4)]
    tue = [pd.Timestamp("2026-01-06 10:00", tz="UTC") + pd.Timedelta(days=7 * k) for k in range(4)]
    train = pd.DataFrame({"y": [1, 1, 0, 1, 0, 0, 0, 0], "minutes_since_last_end": 100.0}, index=pd.DatetimeIndex(mon + tue))
    test = pd.DataFrame({"y": [0], "minutes_since_last_end": 100.0}, index=pd.DatetimeIndex([mon[0] + pd.Timedelta(days=28)]))
    assert hour_of_week_rate(train, test, 3).tolist() == [0.75]


def test_recent_activity_splits_by_minutes_since_end():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    train = pd.DataFrame({"y": [1, 1, 0, 0], "minutes_since_last_end": [10, 20, 500, 600.0]}, index=idx)
    test = pd.DataFrame({"y": 0, "minutes_since_last_end": [30, 400.0]}, index=idx[:2])
    assert recent_activity_rate(train, test, 3).tolist() == [1.0, 0.0]  # 3 h = 180 min


def test_metrics_known_values():
    y, p = np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.3])
    assert abs(brier(y, p) - np.mean([0.01, 0.01, 0.04, 0.09])) < 1e-12
    assert pr_auc(y, p) == 1.0
    table = reliability(np.tile(y, 5), np.tile(p, 5), bins=2)
    assert table["n"].tolist() == [10, 10]


def test_bootstrap_resamples_whole_days_and_is_reproducible():
    rng = np.random.default_rng(3)
    n = 40 * 20
    day = np.repeat([f"2026-01-{d:02d}" for d in range(1, 41)], 20)
    y = (rng.random(n) < 0.5).astype(int)
    res = pd.DataFrame({"y": y, "day": day, "good": 0.2 + 0.6 * y, "flat": np.full(n, 0.5)})
    t1, d1 = day_block_bootstrap(res, ["good", "flat"], reference="flat", n_boot=200, seed=1)
    t2, d2 = day_block_bootstrap(res, ["good", "flat"], reference="flat", n_boot=200, seed=1)
    assert t1.equals(t2) and d1.equals(d2)
    brier_good = t1[(t1.model == "good") & (t1.metric == "brier")].iloc[0]
    assert brier_good.lo <= brier_good.value <= brier_good.hi
    diff = d1[d1.metric == "brier"].iloc[0]
    assert diff.hi < 0 and diff.share_better == 1.0  # a perfect-ish model clearly beats a flat one


def test_smoothed_hour_of_week_moves_from_slot_mean_to_base_rate():
    from alerts_forecast.baselines import make_smoothed_hour_of_week

    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-05 10:00", tz="UTC") + pd.Timedelta(days=7 * k) for k in range(4)]
        + [pd.Timestamp("2026-01-06 10:00", tz="UTC") + pd.Timedelta(days=7 * k) for k in range(4)]
    )
    train = pd.DataFrame({"y": [1, 1, 1, 1, 0, 0, 0, 0], "minutes_since_last_end": 100.0}, index=idx)
    test = pd.DataFrame({"y": 0, "minutes_since_last_end": 100.0}, index=idx[:1] + pd.Timedelta(days=28))
    assert make_smoothed_hour_of_week(0)(train, test, 3).tolist() == [1.0]  # no smoothing: slot mean
    assert abs(make_smoothed_hour_of_week(4)(train, test, 3)[0] - (4 + 4 * 0.5) / 8) < 1e-12  # (sum+k*base)/(n+k)
    assert abs(make_smoothed_hour_of_week(1e9)(train, test, 3)[0] - 0.5) < 1e-6


def test_logreg_fit_predict_uses_only_train_rows_for_fitting():
    from alerts_forecast.models import make_logreg

    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    df = pd.DataFrame({"x": x, "y": (x + rng.normal(scale=0.5, size=400) > 0)})
    train, test = df.iloc[:300], df.iloc[300:]
    p1 = make_logreg(["x"], 1.0)(train, test, 3)
    flipped = test.assign(y=~test["y"])
    assert np.allclose(p1, make_logreg(["x"], 1.0)(train, flipped, 3))  # test labels are ignored
    assert np.corrcoef(p1, test["x"])[0, 1] > 0.9 and p1.min() >= 0 and p1.max() <= 1


def test_recent_level_baseline_uses_only_the_last_days_before_each_day():
    from alerts_forecast.experiment import RECENT_LEVEL, run_recent_level

    idx = pd.date_range("2026-01-01", periods=40 * 96, freq="15min", tz="UTC")
    day = np.arange(len(idx)) // 96
    sample = pd.DataFrame({"y": (day >= 20).astype(int), "minutes_since_last_end": 100.0}, index=idx)  # level jumps on day 20
    start, end = pd.Timestamp("2026-01-31", tz="UTC"), pd.Timestamp("2026-02-05", tz="UTC")
    out = run_recent_level(sample, 3, start, end)
    assert len(out) > 0 and {f"{RECENT_LEVEL}||3d", f"{RECENT_LEVEL}||30d"} <= set(out.columns)
    # long after the jump the 3-day window knows the new level, the 30-day window is still mixed
    assert np.allclose(out[f"{RECENT_LEVEL}||3d"], 1.0) and out[f"{RECENT_LEVEL}||30d"].between(0.2, 0.6).all()
    # purge: the training rows used for a day end at least H hours before that day starts
    first = out.index.min()
    assert out.loc[first, f"{RECENT_LEVEL}||3d"] == sample.loc[(sample.index > first - pd.Timedelta(hours=3, days=3)) & (sample.index <= first - pd.Timedelta(hours=3)), "y"].mean()


def test_feature_grid_covers_every_horizon_and_missing_features_are_refused():
    from alerts_forecast.experiment import feature_index, join_features
    from alerts_forecast.target import build_frame, main_sample

    start = pd.Timestamp("2026-01-01 10:00", tz="UTC").as_unit("ns")
    alerts = pd.DataFrame({
        "started_at": [start + pd.Timedelta(hours=k * 5) for k in range(10)],
        "finished_at": [start + pd.Timedelta(hours=k * 5, minutes=40) for k in range(10)],
        "naive": False,
    })
    data_end = start + pd.Timedelta(hours=60)
    horizons = [3, 1, 0.25]
    idx = feature_index(alerts, data_end, horizons)
    feats = pd.DataFrame({"f": 1.0}, index=idx)
    for h in horizons:
        sample = main_sample(build_frame(alerts, data_end, h))
        assert sample.index.isin(idx).all(), h  # the shortest horizon reaches furthest
        assert not join_features(sample, feats)["f"].isna().any()
    # the old way (features on the 1 h grid) leaves the last 45 minutes of the 15-minute sample without features
    old = pd.DataFrame({"f": 1.0}, index=feature_index(alerts, data_end, [1]))
    with pytest.raises(ValueError, match="without features"):
        join_features(main_sample(build_frame(alerts, data_end, 0.25)), old)


def test_block_units_for_the_bootstrap():
    from alerts_forecast.metrics import with_blocks

    idx = pd.date_range("2026-03-01 20:00", periods=8, freq="3h", tz="UTC")  # Kyiv = UTC+2 before 2026-03-29
    res = pd.DataFrame({"y": 0, "day": "x"}, index=idx)
    assert with_blocks(res, "day")["day"].iloc[0] == "2026-03-01"  # 22:00 Kyiv
    assert with_blocks(res, "day")["day"].iloc[1] == "2026-03-02"  # 01:00 Kyiv
    weeks = with_blocks(res, "week")["day"]
    assert weeks.iloc[0] == "2026-09" and weeks.iloc[1] == "2026-10"  # Sunday 22:00 vs Monday 01:00 (ISO weeks)
    six = with_blocks(res, "6h")["day"].tolist()
    assert six[:3] == ["2026-03-01-3", "2026-03-02-0", "2026-03-02-0"]  # 22:00, 01:00, 04:00 Kyiv


def test_evaluate_purges_validation_and_picks_strict_configurations():
    from alerts_forecast.baselines import constant_rate, hour_of_week_rate, recent_activity_rate
    from alerts_forecast.experiment import STRICT_BRIER, STRICT_PR, evaluate
    from alerts_forecast.metrics import brier, pr_auc
    from alerts_forecast.models import make_logreg

    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=120 * 96, freq="15min", tz="UTC")
    x = rng.normal(size=len(idx))
    sample = pd.DataFrame({"y": rng.random(len(idx)) < 1 / (1 + np.exp(-x)), "x": x,
                           "minutes_since_last_end": rng.uniform(0, 600, len(idx)), "prev_naive": False}, index=idx)
    cfg = [("constant", "", "90d", constant_rate), ("recent_activity", "", "90d", recent_activity_rate),
           ("hour_of_week", "", "90d", hour_of_week_rate), ("logreg[x]", "C=1", "90d", make_logreg(["x"], 1.0))]
    h, test_start = 3, pd.Timestamp("2026-04-01", tz="UTC")
    ev = evaluate(sample, h, test_start - pd.Timedelta(weeks=4), test_start, test_start + pd.Timedelta(weeks=2), cfg)
    assert ev.val.index.max() + pd.Timedelta(hours=h) <= test_start  # validation labels stay out of the test block
    assert ev.lr_best == "logreg[x]" and ev.hgb_best == ""
    base = [c for c in ev.test.columns if "|" in c and not c.startswith("logreg")]
    assert ev.oracle == min(base, key=lambda c: brier(ev.test["y"], ev.test[c]))
    assert ev.oracle_pr == max(base, key=lambda c: pr_auc(ev.test["y"], ev.test[c]))
    assert np.allclose(ev.res[STRICT_BRIER], ev.test[ev.oracle]) and np.allclose(ev.res[STRICT_PR], ev.test[ev.oracle_pr])
