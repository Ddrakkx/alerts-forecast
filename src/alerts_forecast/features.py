"""Features at forecast moment t. Every value uses only alerts with started_at <= t.

Ends of alerts are used in two ways only: "is the alert still running at t" and the time already
spent under alert up to t. Nothing like "time left" exists. tests/test_features.py cuts the data at t
and checks that the values do not change.
"""
import numpy as np
import pandas as pd

from .target import _ns, state_at

KYIV = "Europe/Kyiv"
# Neighbouring oblasts, from my knowledge of the map, NOT from the data (the dataset has no geometry). To be checked on a map.
# Luhanska oblast is not in the data (permanent siren). "Kyiv City" is an enclave inside Kyivska oblast.
NEIGHBORS_BY_REGION = {
    "Poltavska oblast": ("Chernihivska oblast", "Sumska oblast", "Kharkivska oblast", "Dnipropetrovska oblast",
                         "Kirovohradska oblast", "Cherkaska oblast", "Kyivska oblast"),
    "Kyivska oblast": ("Zhytomyrska oblast", "Chernihivska oblast", "Poltavska oblast", "Cherkaska oblast",
                       "Vinnytska oblast", "Kyiv City"),
    "Kharkivska oblast": ("Sumska oblast", "Poltavska oblast", "Dnipropetrovska oblast", "Donetska oblast"),
    "Lvivska oblast": ("Volynska oblast", "Rivnenska oblast", "Ternopilska oblast", "Ivano-Frankivska oblast",
                       "Zakarpatska oblast"),
}
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


def neighbors_of(region: str) -> tuple:
    if region not in NEIGHBORS_BY_REGION:
        raise KeyError(f"no neighbour list for {region!r}; add it to NEIGHBORS_BY_REGION")
    return NEIGHBORS_BY_REGION[region]


def nbr_column(name: str) -> str:
    return f"nbr_active_{name.split()[0].lower()}"


def build_features(by_region: dict, target: str, times, neighbors=None) -> pd.DataFrame:
    """Own, neighbour and country-wide features for the given moments (a tz-aware DatetimeIndex)."""
    neighbors = neighbors_of(target) if neighbors is None else neighbors
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
    for name in neighbors:
        a = state_at(by_region[name], times)["active"].to_numpy()
        f[nbr_column(name)] = a.astype(float)
        nbr_active.append(a)
    f["nbr_active_n"] = np.sum(nbr_active, axis=0).astype(float)
    for name, w in (("1h", pd.Timedelta(hours=1)), ("3h", pd.Timedelta(hours=3))):
        f[f"nbr_starts_{name}"] = sum(count_starts(by_region[n], t, w) for n in neighbors)
    since_start = np.min([minutes_since_last_start(by_region[n], t, 1440.0) for n in neighbors], axis=0)
    f["nbr_log_since_start"] = np.log1p(since_start)

    others = [r for r in by_region if r != target]
    f["cty_active_n"] = np.sum([state_at(by_region[r], times)["active"].to_numpy() for r in others], axis=0).astype(float)
    for name, w in (("1h", pd.Timedelta(hours=1)), ("3h", pd.Timedelta(hours=3)), ("24h", pd.Timedelta(hours=24))):
        f[f"cty_starts_{name}"] = sum(count_starts(by_region[r], t, w) for r in others)
    return pd.DataFrame(f, index=times)


_OWN = ["own_log_since_end", "own_starts_3h", "own_starts_24h", "own_starts_7d", "own_active_frac_24h",
        "tod_sin1", "tod_cos1", "tod_sin2", "tod_cos2", "is_weekend"]
_CTY = ["cty_active_n", "cty_starts_1h", "cty_starts_3h", "cty_starts_24h"]


def feature_sets(neighbors) -> dict:
    nbr = [nbr_column(n) for n in neighbors] + ["nbr_active_n", "nbr_starts_1h", "nbr_starts_3h", "nbr_log_since_start"]
    return {"own": _OWN, "own+nbr": _OWN + nbr, "own+nbr+cty": _OWN + nbr + _CTY}
