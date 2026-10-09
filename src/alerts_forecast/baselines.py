"""Baselines. Each is fit_predict(train, test, horizon_h) -> probabilities for the test rows.

train/test are slices of the main sample (no active alert at t). Only train labels are used.
"""
import numpy as np
import pandas as pd

KYIV = "Europe/Kyiv"


def kyiv_hour_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """0..167 = weekday * 24 + hour on the Kyiv wall clock (follows summer/winter time)."""
    local = index.tz_convert(KYIV)
    return (local.dayofweek * 24 + local.hour).to_numpy()


def constant_rate(train: pd.DataFrame, test: pd.DataFrame, horizon_h: int) -> np.ndarray:
    """Reference: always the base rate of the training window."""
    return np.full(len(test), train["y"].mean())


def hour_of_week_rate(train: pd.DataFrame, test: pd.DataFrame, horizon_h: int) -> np.ndarray:
    """Mean label of the same hour of the week in the training window."""
    rates = pd.Series(train["y"].to_numpy(float)).groupby(kyiv_hour_of_week(train.index)).mean()
    mapped = pd.Series(kyiv_hour_of_week(test.index)).map(rates)
    return mapped.fillna(train["y"].mean()).to_numpy()


def make_smoothed_hour_of_week(k: float):
    """POST-HOC baseline (added after the first test results were seen).

    Slot mean pulled to the training base rate: (sum_y + k * base) / (n + k). Rows of one slot are
    strongly correlated (4 per hour, one day at a time), so the useful k is large; chosen on validation.
    """

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, horizon_h: int) -> np.ndarray:
        base = train["y"].mean()
        stats = pd.Series(train["y"].to_numpy(float)).groupby(kyiv_hour_of_week(train.index)).agg(["sum", "count"])
        slot_rate = (stats["sum"] + k * base) / (stats["count"] + k)
        return pd.Series(kyiv_hour_of_week(test.index)).map(slot_rate).fillna(base).to_numpy()

    return fit_predict


def recent_activity_rate(train: pd.DataFrame, test: pd.DataFrame, horizon_h: int) -> np.ndarray:
    """Persistence: label rate after 'an alert ended within the last H hours' vs otherwise.

    The main sample has no active alert at t, so plain "as now" would always say 0.
    """
    base = train["y"].mean()
    near_tr = (train["minutes_since_last_end"] <= horizon_h * 60).to_numpy()
    near_te = (test["minutes_since_last_end"] <= horizon_h * 60).to_numpy()
    rate_near = train["y"][near_tr].mean() if near_tr.any() else base
    rate_far = train["y"][~near_tr].mean() if (~near_tr).any() else base
    return np.where(near_te, rate_near, rate_far)


BASELINES = {
    "constant": constant_rate,
    "recent_activity": recent_activity_rate,
    "hour_of_week": hour_of_week_rate,
}
