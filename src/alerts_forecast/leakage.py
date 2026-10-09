"""Leakage check: a feature function must not change when the future is removed."""
import pandas as pd

# stands for "end not known yet" in a table cut at time t
UNKNOWN_END = pd.Timestamp("2200-01-01", tz="UTC").as_unit("ns")


def truncate_at(alerts: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """The table as it looked at time t: later alerts are absent, ends after t are not known yet."""
    known = alerts.loc[alerts["started_at"] <= t].copy()
    known.loc[known["finished_at"] > t, "finished_at"] = UNKNOWN_END
    return known


def truncation_violations(fn, alerts: pd.DataFrame, times) -> list:
    """Moments t where fn(full table, [t]) differs from fn(table cut at t, [t]).

    fn(alerts, times) must return a DataFrame with one row per time. An empty result means
    that fn at t did not use anything that happened after t.
    """
    bad = []
    for t in times:
        idx = pd.DatetimeIndex([t])
        full = fn(alerts, idx)
        cut = fn(truncate_at(alerts, t), idx)
        if not full.equals(cut):
            bad.append(t)
    return bad
