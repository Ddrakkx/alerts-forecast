import numpy as np
import pandas as pd
import pytest

from alerts_forecast.data import load_official_regions, merge_intervals
from alerts_forecast.metrics import calibration_slope_intercept
from alerts_forecast.models import make_logreg, make_platt


def iv(rows) -> pd.DataFrame:
    f = lambda s: pd.Timestamp(f"2026-01-01 {s}", tz="UTC").as_unit("ns")  # noqa: E731
    return pd.DataFrame({"started_at": [f(a) for a, _ in rows], "finished_at": [f(b) for _, b in rows]})


def test_merge_intervals_joins_overlapping_and_touching_but_not_separate():
    out = merge_intervals(iv([("10:00", "10:30"), ("10:20", "11:00"), ("11:00", "11:10"), ("12:00", "12:10"), ("11:50", "12:05")]))
    got = [(a.strftime("%H:%M"), b.strftime("%H:%M")) for a, b in zip(out.started_at, out.finished_at)]
    assert got == [("10:00", "11:10"), ("11:50", "12:10")]


def test_merge_keeps_a_long_alert_that_swallows_later_short_ones():
    out = merge_intervals(iv([("10:00", "15:00"), ("11:00", "11:10"), ("12:00", "12:10"), ("16:00", "16:10")]))
    assert len(out) == 2 and out.finished_at.iloc[0].strftime("%H:%M") == "15:00"


def test_official_loader_drops_duplicates_and_merges_parts_of_an_oblast(tmp_path):
    head = "oblast,raion,hromada,level,started_at,finished_at,source\n"
    rows = [
        "A oblast,,,oblast,2026-01-01 10:00:00+00:00,2026-01-01 10:30:00+00:00,official",
        "A oblast,,,oblast,2026-01-01 10:00:00+00:00,2026-01-01 10:30:00+00:00,official",  # exact duplicate
        "A oblast,X raion,,raion,2026-01-01 10:20:00+00:00,2026-01-01 11:00:00+00:00,official",
        "A oblast,Y raion,,raion,2026-01-01 13:00:00+00:00,2026-01-01 13:10:00+00:00,official",
        "Luhanska oblast,,,oblast,2026-01-01 10:00:00+00:00,2026-01-01 10:30:00+00:00,official",
    ]
    p = tmp_path / "o.csv"
    p.write_text(head + "\n".join(rows) + "\n")
    out = load_official_regions(p)
    assert list(out) == ["A oblast"]
    assert len(out["A oblast"]) == 2 and not out["A oblast"]["naive"].any()
    assert out["A oblast"]["finished_at"].iloc[0].strftime("%H:%M") == "11:00"


def test_calibration_slope_of_perfect_and_overconfident_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.random(20000) < p
    slope, icpt = calibration_slope_intercept(y, p)
    assert abs(slope - 1) < 0.06 and abs(icpt) < 0.06
    logit = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2.5 * logit))  # pushed away from 0.5
    assert calibration_slope_intercept(y, over)[0] < 0.6


def _toy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"x": x, "y": rng.random(n) < 1 / (1 + np.exp(-2 * x))}, index=idx)


def test_platt_ignores_test_labels_and_keeps_a_gap_between_fit_and_calibration():
    df = _toy()
    train, test = df.iloc[:1200], df.iloc[1200:]
    seen = {}

    def spy(fit, pred, h):
        seen["fit_end"], seen["pred_start"] = fit.index.max(), pred.index.min()
        return make_logreg(["x"], 1.0)(fit, pred, h)

    p1 = make_platt(spy, cal_days=10)(train, test, 6)
    p2 = make_platt(make_logreg(["x"], 1.0), cal_days=10)(train, test.assign(y=~test["y"]), 6)
    assert np.allclose(p1, p2)
    # the model is fit on rows at least cal_days + H hours before the end of training
    assert seen["fit_end"] <= train.index.max() - pd.Timedelta(days=10) - pd.Timedelta(hours=6)
    assert seen["pred_start"] > seen["fit_end"]


def test_platt_repairs_an_overconfident_model():
    df = _toy(n=4000)
    train, test = df.iloc[:3000], df.iloc[3000:]

    def overconfident(fit, pred, h):
        z = 5 * pred["x"].to_numpy()  # true slope is 2
        return 1 / (1 + np.exp(-z))

    before = calibration_slope_intercept(test["y"], overconfident(train, test, 3))[0]
    after = calibration_slope_intercept(test["y"], make_platt(overconfident, cal_days=60)(train, test, 3))[0]
    assert before < 0.6 and abs(after - 1) < 0.2


def test_hgb_is_deterministic_and_ignores_test_labels():
    from alerts_forecast.models import make_hgb

    df = _toy(n=1500)
    train, test = df.iloc[:1200], df.iloc[1200:]
    fn = make_hgb(["x"], 2, 30)
    p1 = fn(train, test, 3)
    assert np.allclose(p1, fn(train, test.assign(y=~test["y"]), 3))
    assert np.allclose(p1, make_hgb(["x"], 2, 30)(train, test, 3))
    assert np.corrcoef(p1, test["x"])[0, 1] > 0.8 and 0 < p1.min() and p1.max() < 1


def test_models_survive_a_training_window_with_one_class():
    from alerts_forecast.models import make_hgb

    df = _toy(n=600)
    train, test = df.iloc[:400].assign(y=False), df.iloc[400:]
    for fn in (make_logreg(["x"], 1.0), make_hgb(["x"], 2, 20)):
        assert fn(train, test, 3).tolist() == [0.0] * len(test)
    # Platt falls back to the raw probabilities when the newest cal_days contain one class only
    h, cal_days = 3, 10
    full = df.iloc[:500].copy()
    cal_start = full.index.max() - pd.Timedelta(days=cal_days)
    full.loc[full.index > cal_start, "y"] = False  # the calibration part is single-class
    fit_part = full.loc[full.index <= cal_start - pd.Timedelta(hours=h)]
    expected = make_logreg(["x"], 1.0)(fit_part, test, h)
    assert np.allclose(make_platt(make_logreg(["x"], 1.0), cal_days=cal_days)(full, test, h), expected)
