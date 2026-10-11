"""Is there less alert activity at weekends than on weekdays? A check of the claim on the volunteer snapshot (to 2026-10-09).

Days are Kyiv calendar days, weekend = Saturday and Sunday. Measures per oblast and for the whole country (all oblasts summed):
alert starts per day, hours under alert per day, and the share of quiet 15-min moments followed by a new alert within 3 h
(the project's main target). Intensity changes a lot from month to month, so weekend and weekdays are compared WITHIN each
Mon-Sun week (a plain mean over all days would mix trends with the weekday effect); the interval is a bootstrap over weeks.

    python analysis/weekend_check.py > results/weekend_check.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import load_regions  # noqa: E402
from alerts_forecast.target import build_frame, main_sample  # noqa: E402

KYIV = "Europe/Kyiv"
OBLASTS = ["Poltavska oblast", "Kyivska oblast", "Kharkivska oblast", "Lvivska oblast"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
N_BOOT, SEED = 2000, 0


def daily(alerts: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Starts and hours under alert per Kyiv calendar day; days are tz-aware Kyiv midnights."""
    starts = alerts["started_at"].dt.tz_convert(KYIV).dt.normalize().value_counts().reindex(days, fill_value=0)
    hours = pd.Series(0.0, index=days)
    for a, b in zip(alerts["started_at"].dt.tz_convert(KYIV), alerts["finished_at"].dt.tz_convert(KYIV)):
        d = a.normalize()
        while d < b:
            nxt = d + pd.DateOffset(days=1)  # a calendar day, also on 23 h and 25 h DST days
            if d in hours.index:
                hours[d] += (min(b, nxt) - max(a, d)).total_seconds() / 3600
            d = nxt
    return pd.DataFrame({"starts": starts.astype(float), "hours": hours})


def onset_share(alerts: pd.DataFrame, data_end: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Series:
    """Per Kyiv day: share of quiet grid moments followed by an alert start within 3 h (NaN on a day without quiet moments)."""
    s = main_sample(build_frame(alerts, data_end, 3))
    day = s.index.tz_convert(KYIV).normalize()
    return s["y"].astype(float).groupby(day).mean().reindex(days)


def weekly_compare(x: pd.Series, rng: np.random.Generator) -> tuple:
    """(weekday mean, weekend mean, ratio weekend/weekday with 95% interval, share of weeks with a quieter weekend).
    Each week contributes its own weekday mean and weekend mean, so a busy month cancels out in the comparison."""
    local = x.index.tz_localize(None)
    df = pd.DataFrame({"x": x.to_numpy(), "week": local - pd.to_timedelta(local.dayofweek, unit="D"), "we": local.dayofweek >= 5})
    w = df.groupby(["week", "we"])["x"].mean().unstack().dropna()
    wd, we = w[False].to_numpy(), w[True].to_numpy()
    idx = rng.integers(0, len(w), size=(N_BOOT, len(w)))
    boot = we[idx].mean(axis=1) / wd[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return wd.mean(), we.mean(), we.mean() / wd.mean(), lo, hi, float((we < wd).mean()), len(w)


def fmt(name: str, r: tuple, unit: str) -> str:
    wd, we, ratio, lo, hi, quieter, n = r
    verdict = "weekend LOWER" if hi < 1 else "weekend HIGHER" if lo > 1 else "no clear difference"
    return (f"  {name:20s} weekday {wd:7.2f} {unit}, weekend {we:7.2f} {unit}, ratio {ratio:.2f} [{lo:.2f}, {hi:.2f}]"
            f"  quieter weekend in {quieter:.0%} of {n} weeks  -> {verdict}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    first = pd.Timestamp("2022-02-28", tz=KYIV)  # the first Monday with complete data
    last_full_day = (data_end.tz_convert(KYIV).normalize() - pd.DateOffset(days=1))
    last = last_full_day - pd.DateOffset(days=(last_full_day.dayofweek + 1) % 7)  # the last complete week ends on a Sunday
    days = pd.date_range(first, last, freq="D")
    per = {r: daily(a, days) for r, a in regions.items()}
    country = sum(per.values())
    periods = {"all (2022-02-28..)": days, "last 52 weeks": days[-52 * 7:], "last 8 weeks (test block)": days[-8 * 7:]}

    print(f"Volunteer snapshot, data end {data_end}; complete Mon-Sun weeks {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d} (Kyiv days).")
    print("Ratio = weekend daily mean / weekday daily mean, both taken within the same weeks; 95% bootstrap over weeks "
          f"({N_BOOT} draws, seed {SEED}). Ratio < 1: less at weekends.\n")
    for measure, unit in (("starts", "starts/day"), ("hours", "h/day")):
        print(f"== {measure} per day ==")
        for pname, pdays in periods.items():
            print(f" {pname}")
            print(fmt("country (sum)", weekly_compare(country.loc[pdays, measure], rng), unit))
            for r in OBLASTS:
                print(fmt(r, weekly_compare(per[r].loc[pdays, measure], rng), unit))
        print()

    print("== country starts per day, by year (stability of the effect) ==")
    for year in sorted(set(days.year)):
        ydays = days[days.year == year]
        print(fmt(str(year), weekly_compare(country.loc[ydays, "starts"], rng), "starts/day"))
    print()

    print("== starts by day of week, last 52 weeks (index: 1.00 = the mean day of the same weeks) ==")
    print("  " + " " * 20 + "".join(f"{d:>7s}" for d in DAY_NAMES))
    for name, s in [("country (sum)", country["starts"]), *[(r, per[r]["starts"]) for r in OBLASTS]]:
        x = s.loc[periods["last 52 weeks"]]
        prof = x.groupby(x.index.dayofweek).mean() / x.mean()
        print(f"  {name:20s}" + "".join(f"{v:7.2f}" for v in prof))
    print()

    print("== main target: share of quiet moments followed by an alert within 3 h ==")
    shares = {r: onset_share(regions[r], data_end, days) for r in OBLASTS}
    for pname, pdays in periods.items():
        print(f" {pname}")
        for r in OBLASTS:
            print(fmt(r, weekly_compare(shares[r].loc[pdays].dropna(), rng), "share"))


if __name__ == "__main__":
    main()
