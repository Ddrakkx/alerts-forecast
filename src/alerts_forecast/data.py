"""Loading and validation of the volunteer air-raid alert data."""
from pathlib import Path

import pandas as pd

REGION = "Poltavska oblast"
# permanent siren since 2022 (see the dataset README): 3 records, not a real series
EXCLUDED_REGIONS = ("Luhanska oblast",)
REQUIRED_COLUMNS = ("region", "started_at", "finished_at", "naive")


def _read(path) -> tuple[pd.DataFrame, pd.Timestamp]:
    raw = pd.read_csv(Path(path))
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for col in ("started_at", "finished_at"):
        if not raw[col].str.endswith("+00:00").all():
            raise ValueError(f"{col}: found timestamps that are not in UTC")
        raw[col] = pd.to_datetime(raw[col], utc=True).dt.as_unit("ns")
    data_end = max(raw["started_at"].max(), raw["finished_at"].max())
    return raw, data_end


def _region_alerts(raw: pd.DataFrame, region: str) -> pd.DataFrame:
    alerts = raw.loc[raw["region"] == region, ["started_at", "finished_at", "naive"]]
    alerts = alerts.sort_values("started_at").reset_index(drop=True)
    if alerts.empty:
        raise ValueError(f"no alerts for region {region!r}")
    validate_alerts(alerts)
    return alerts


def load_volunteer(path, region: str = REGION) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return (alerts of one region, end of data).

    alerts: columns started_at, finished_at (UTC, ns), naive (end was invented), sorted by start.
    end of data: the latest timestamp observed in the WHOLE file (any oblast, start or end).
    A region without alerts after its last event is only known to be quiet up to this moment.
    """
    raw, data_end = _read(path)
    return _region_alerts(raw, region), data_end


def load_regions(path, exclude=EXCLUDED_REGIONS) -> tuple[dict, pd.Timestamp]:
    """Return ({region: alerts}, end of data) for every region except the excluded ones."""
    raw, data_end = _read(path)
    return {r: _region_alerts(raw, r) for r in sorted(raw["region"].unique()) if r not in exclude}, data_end


def merge_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Union of intervals: overlapping or touching ones become one episode."""
    d = df.sort_values("started_at")
    run_end = d["finished_at"].cummax().shift()
    group = (d["started_at"] > run_end).cumsum()  # NaT for the first row compares False, cumsum starts the first group at 0
    out = d.groupby(group).agg(started_at=("started_at", "min"), finished_at=("finished_at", "max"))
    return out.reset_index(drop=True)


def load_official_regions(path, exclude=EXCLUDED_REGIONS) -> dict:
    """Official file as {oblast: episodes}: exact duplicates dropped, raion/hromada alerts merged into
    oblast-level episodes (an oblast is under alert while any of its parts is). naive is always False."""
    raw = pd.read_csv(Path(path))
    for col in ("started_at", "finished_at"):
        if not raw[col].str.endswith("+00:00").all():
            raise ValueError(f"{col}: found timestamps that are not in UTC")
        raw[col] = pd.to_datetime(raw[col], utc=True).dt.as_unit("ns")
    raw = raw.drop_duplicates()
    out = {}
    for oblast, d in raw.groupby("oblast"):
        if oblast in exclude:
            continue
        episodes = merge_intervals(d[["started_at", "finished_at"]])
        episodes["naive"] = False
        validate_alerts(episodes)
        out[oblast] = episodes
    return out


def validate_alerts(alerts: pd.DataFrame) -> None:
    """Alerts of one region must be sorted, end after start and must not overlap."""
    if (alerts["finished_at"] < alerts["started_at"]).any():
        raise ValueError("found an alert that ends before it starts")
    if not alerts["started_at"].is_monotonic_increasing:
        raise ValueError("alerts are not sorted by start")
    if (alerts["started_at"].iloc[1:].to_numpy() < alerts["finished_at"].iloc[:-1].to_numpy()).any():
        raise ValueError("found overlapping alerts in one region")
