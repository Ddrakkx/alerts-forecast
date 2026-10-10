"""Share of positive labels per month (main sample), next to alert counts and durations.

    python scripts/target_by_month.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import REGION, load_volunteer  # noqa: E402
from alerts_forecast.target import HORIZONS_H, build_frame, main_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=REGION)
    alerts, data_end = load_volunteer(ROOT / "data" / "raw" / "volunteer_data_en.csv", ap.parse_args().region)
    month = lambda idx: idx.tz_convert("Europe/Kyiv").strftime("%Y-%m")  # noqa: E731
    table = pd.DataFrame(index=sorted(set(month(pd.DatetimeIndex(alerts["started_at"])))))
    table["alerts"] = pd.Series(1, index=pd.DatetimeIndex(alerts["started_at"])).groupby(month).sum()
    dur = pd.Series(
        ((alerts["finished_at"] - alerts["started_at"]) / pd.Timedelta(minutes=1)).to_numpy(),
        index=pd.DatetimeIndex(alerts["started_at"]),
    )
    table["median_min"] = dur.groupby(month).median().round(0)
    for h in HORIZONS_H:
        frame = build_frame(alerts, data_end, h)
        main = main_sample(frame)
        table["active%"] = (frame["active"].groupby(month(frame.index)).mean() * 100).round(1)
        table[f"pos%_H{h}"] = (main["y"].groupby(month(main.index)).mean() * 100).round(1)
    print(table.to_string())


if __name__ == "__main__":
    main()
