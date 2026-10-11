"""The numbers of the README Data section and the Lviv lead-time remark, reproduced from the pinned snapshot (decision 15).

    python analysis/data_checks.py > results/data_checks.txt
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import load_official_regions, load_regions  # noqa: E402
from alerts_forecast.features import neighbors_of  # noqa: E402

WEST = ["Lvivska oblast", "Volynska oblast", "Rivnenska oblast", "Khmelnytska oblast", "Chernivetska oblast", "Ternopilska oblast",
        "Ivano-Frankivska oblast", "Zakarpatska oblast"]
CHECK = (pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-08-29", tz="UTC"))  # May - Aug 28, before the official hole


def inside_any(starts: pd.Series, episodes: pd.DataFrame) -> pd.Series:
    """True where a start lies inside an episode [start, end)."""
    return starts.apply(lambda s: bool(((episodes["started_at"] <= s) & (s < episodes["finished_at"])).any()))


def main() -> None:
    vol, end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    off = load_official_regions(ROOT / "data" / "raw" / "official_data_en.csv")
    test_start = end.floor("D") - pd.Timedelta(weeks=8)
    in_test = lambda d: d[(d["started_at"] >= test_start) & (d["started_at"] <= end)]  # noqa: E731
    in_check = lambda d: d[(d["started_at"] >= CHECK[0]) & (d["started_at"] < CHECK[1])]  # noqa: E731
    print(f"volunteer data end {end}; test block from {test_start:%Y-%m-%d %H:%M} UTC; check period {CHECK[0]:%Y-%m-%d} .. {CHECK[1]:%Y-%m-%d} (exclusive), UTC\n")

    p = vol["Poltavska oblast"]
    print(f"Invented ends (naive), Poltava: whole period {p['naive'].mean():.1%}; test block {int(in_test(p)['naive'].sum())} of {len(in_test(p))} "
          f"({in_test(p)['naive'].mean():.1%})")
    for r in ("Rivnenska oblast", "Ternopilska oblast", "Volynska oblast"):
        d = in_test(vol[r])
        print(f"  Lviv neighbour {r}, test block: naive {int(d['naive'].sum())} of {len(d)}")

    print("\nVolunteer starts outside any official episode of the same oblast (official: deduplicated, raions merged), check period:")
    for r in ("Poltavska oblast", "Kyivska oblast", "Lvivska oblast"):
        v = in_check(vol[r])
        out = ~inside_any(v["started_at"], off[r])
        print(f"  {r}: {int(out.sum())} of {len(v)} ({out.mean():.0%}); official episodes starting in the period: {len(in_check(off[r]))}")

    print("\nAlerts in the test block, western oblasts: " + ", ".join(f"{r.split()[0]} {len(in_test(vol[r]))}" for r in WEST))

    print("\nOwn alert starts in the test block with a neighbour's start shortly before (within m minutes, strictly before):")
    for r in ("Lvivska oblast", "Poltavska oblast"):
        own = in_test(vol[r])["started_at"]
        nb = pd.concat([vol[n]["started_at"] for n in neighbors_of(r)])
        shares = [own.apply(lambda s: bool(((nb < s) & (nb >= s - pd.Timedelta(minutes=m))).any())).mean() for m in (2, 15)]
        print(f"  {r}: 2 min {shares[0]:.0%}, 15 min {shares[1]:.0%}")


if __name__ == "__main__":
    main()
