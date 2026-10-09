"""Features at forecast moment t. Every value uses only alerts with started_at <= t.

Ends of alerts are used in two ways only: "is the alert still running at t" and the time already
spent under alert up to t. Nothing like "time left" exists. tests/test_features.py cuts the data at t
and checks that the values do not change.
"""
import numpy as np
import pandas as pd

from .target import _ns, state_at

KYIV = "Europe/Kyiv"
# From my memory of the map, NOT from the data (no geometry in the dataset). Check on a map.
NEIGHBORS = (
    "Chernihivska oblast", "Sumska oblast", "Kharkivska oblast", "Dnipropetrovska oblast",
    "Kirovohradska oblast", "Cherkaska oblast", "Kyivska oblast",
)
MIN = np.timedelta64(1, "m")


def _starts(alerts) -> np.ndarray:
    return _ns(alerts["started_at"])


def count_starts(alerts, t: np.ndarray, window: pd.Timedelta) -> np.ndarray:
    """Number of alert starts in (t - window, t]."""
    starts = _starts(alerts)
    w = window.to_timedelta64()
    return np.searchsorted(starts, t, side="right") - np.searchsorted(starts, t - w, side="right")


def active_time_before(alerts, x: np.ndarray) -> np.ndarray:
    """Total minutes under alert before moment x (alerts must not overlap).

    An alert that is still running at x counts only up to x, so its end is never needed.
    """
    if alerts.empty:
        return np.zeros(len(x))
    starts, ends = _ns(alerts["started_at"]), _ns(alerts["finished_at"])
    dur = (ends - starts) / MIN
    before = np.concatenate([[0.0], np.cumsum(dur)])  # total duration of the first k alerts
    i = np.searchsorted(starts, x, side="right") - 1
    k = np.maximum(i, 0)
    partial = np.minimum(np.maximum((x - starts[k]) / MIN, 0.0), dur[k])
    return np.where(i >= 0, before[k] + partial, 0.0)


def minutes_since_last_start(alerts, t: np.ndarray, cap: float) -> np.ndarray:
    if alerts.empty:
        return np.full(len(t), cap)
    starts = _starts(alerts)
    i = np.searchsorted(starts, t, side="right") - 1
    since = (t - starts[np.maximum(i, 0)]) / MIN
    return np.where(i >= 0, np.minimum(since, cap), cap)


def build_features(by_region: dict, target: str, times) -> pd.DataFrame:
    """Own, neighbour and country-wide features for the given moments (a tz-aware DatetimeIndex)."""
    times = pd.DatetimeIndex(times)
    t = _ns(times)
    own = by_region[target]
    hour = pd.Series(times.tz_convert(KYIV))
    tod = (hour.dt.hour + hour.dt.minute / 60).to_numpy() / 24.0
    f = {}

    since_end = state_at(own, times)["minutes_since_last_end"].to_numpy()
    f["own_log_since_end"] = np.log1p(np.nan_to_num(np.minimum(since_end, 7 * 1440), nan=0.0))
    for name, w in (("3h", pd.Timedelta(hours=3)), ("24h", pd.Timedelta(hours=24)), ("7d", pd.Timedelta(days=7))):
        f[f"own_starts_{name}"] = count_starts(own, t, w)
    day = pd.Timedelta(hours=24).to_timedelta64()
    f["own_active_frac_24h"] = (active_time_before(own, t) - active_time_before(own, t - day)) / 1440.0
    for h in (1, 2):
        f[f"tod_sin{h}"] = np.sin(2 * np.pi * h * tod)
        f[f"tod_cos{h}"] = np.cos(2 * np.pi * h * tod)
    f["is_weekend"] = (times.tz_convert(KYIV).dayofweek >= 5).astype(float)

    nbr_active = []
    for name in NEIGHBORS:
        a = state_at(by_region[name], times)["active"].to_numpy()
        f[f"nbr_active_{name.split()[0].lower()}"] = a.astype(float)
        nbr_active.append(a)
    f["nbr_active_n"] = np.sum(nbr_active, axis=0).astype(float)
    for name, w in (("1h", pd.Timedelta(hours=1)), ("3h", pd.Timedelta(hours=3))):
        f[f"nbr_starts_{name}"] = sum(count_starts(by_region[n], t, w) for n in NEIGHBORS)
    since_start = np.min([minutes_since_last_start(by_region[n], t, 1440.0) for n in NEIGHBORS], axis=0)
    f["nbr_log_since_start"] = np.log1p(since_start)

    others = [r for r in by_region if r != target]
    f["cty_active_n"] = np.sum([state_at(by_region[r], times)["active"].to_numpy() for r in others], axis=0).astype(float)
    for name, w in (("1h", pd.Timedelta(hours=1)), ("3h", pd.Timedelta(hours=3)), ("24h", pd.Timedelta(hours=24))):
        f[f"cty_starts_{name}"] = sum(count_starts(by_region[r], t, w) for r in others)
    return pd.DataFrame(f, index=times)


_OWN = ["own_log_since_end", "own_starts_3h", "own_starts_24h", "own_starts_7d", "own_active_frac_24h",
        "tod_sin1", "tod_cos1", "tod_sin2", "tod_cos2", "is_weekend"]
_NBR = [f"nbr_active_{n.split()[0].lower()}" for n in NEIGHBORS] + [
    "nbr_active_n", "nbr_starts_1h", "nbr_starts_3h", "nbr_log_since_start"]
_CTY = ["cty_active_n", "cty_starts_1h", "cty_starts_3h", "cty_starts_24h"]
FEATURE_SETS = {"own": _OWN, "own+nbr": _OWN + _NBR, "own+nbr+cty": _OWN + _NBR + _CTY}
