"""Forecast grid, alert state at time t and the onset label."""
import numpy as np
import pandas as pd

GRID_STEP = pd.Timedelta(minutes=15)
HORIZONS_H = (1, 3, 6)
MAIN_HORIZON_H = 3


def _ns(values) -> np.ndarray:
    """tz-aware datetimes -> naive UTC datetime64[ns] array."""
    return pd.DatetimeIndex(values).tz_convert("UTC").tz_localize(None).as_unit("ns").to_numpy()


def state_at(alerts: pd.DataFrame, times) -> pd.DataFrame:
    """What is known at each t, using only alerts with started_at <= t.

    active: started_at <= t < finished_at of the latest started alert.
    minutes_since_last_end: only when that alert has already ended, else NaN.
    prev_naive: the latest alert has an invented end. For evaluation filters only,
    never a model feature (in real time nobody knows the end was invented).
    """
    index = pd.DatetimeIndex(times)
    if alerts.empty:  # nothing is known yet
        n = len(index)
        return pd.DataFrame(
            {"active": np.zeros(n, bool), "minutes_since_last_end": np.full(n, np.nan), "prev_naive": np.zeros(n, bool)},
            index=index,
        )
    starts, ends = _ns(alerts["started_at"]), _ns(alerts["finished_at"])
    t = _ns(times)
    i = np.searchsorted(starts, t, side="right") - 1  # latest alert started at or before t
    seen = i >= 0
    last = np.maximum(i, 0)
    end = ends[last]
    since = np.where(seen & (end <= t), (t - end) / np.timedelta64(1, "m"), np.nan)
    return pd.DataFrame(
        {
            "active": seen & (end > t),
            "minutes_since_last_end": since,
            "prev_naive": seen & alerts["naive"].to_numpy()[last],
        },
        index=index,
    )


def onset_within(alerts: pd.DataFrame, times, horizon: pd.Timedelta) -> np.ndarray:
    """Label: True if an alert STARTS in the window (t, t + horizon]. A start exactly at t is the past."""
    t = _ns(times)
    if alerts.empty:
        return np.zeros(len(t), bool)
    starts = _ns(alerts["started_at"])
    j = np.searchsorted(starts, t, side="right")  # first start strictly after t
    has_next = j < len(starts)
    next_start = starts[np.minimum(j, len(starts) - 1)]
    return has_next & (next_start <= t + horizon.to_timedelta64())


def make_grid(alerts: pd.DataFrame, data_end: pd.Timestamp, horizon: pd.Timedelta) -> pd.DatetimeIndex:
    """15-min grid from the first alert to the last t whose label window is fully observed."""
    first = alerts["started_at"].iloc[0].ceil(GRID_STEP)
    last = (data_end - horizon).floor(GRID_STEP)
    return pd.date_range(first, last, freq=GRID_STEP)


def build_frame(alerts: pd.DataFrame, data_end: pd.Timestamp, horizon_h: int) -> pd.DataFrame:
    """One row per forecast moment t: state at t plus the label y for the given horizon."""
    horizon = pd.Timedelta(hours=horizon_h)
    times = make_grid(alerts, data_end, horizon)
    frame = state_at(alerts, times)
    frame["y"] = onset_within(alerts, times, horizon)
    frame.index.name = "t"
    return frame


def main_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Main sample: only moments with no active alert."""
    return frame.loc[~frame["active"]]
