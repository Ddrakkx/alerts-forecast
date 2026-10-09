"""Walk-forward evaluation of the baselines on a validation block and the final test block.

The training window (all past vs the last N days) is chosen on the validation block only
(8 weeks before the test block). The test block is looked at once, with the chosen window,
plus an informational table for the other windows.

    python scripts/run_baselines.py [--boot 1000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.baselines import BASELINES  # noqa: E402
from alerts_forecast.data import load_volunteer  # noqa: E402
from alerts_forecast.metrics import brier, day_block_bootstrap, pr_auc, reliability  # noqa: E402
from alerts_forecast.target import HORIZONS_H, MAIN_HORIZON_H, build_frame, main_sample  # noqa: E402
from alerts_forecast.walkforward import walk_forward  # noqa: E402

WINDOWS = {"all past": None, "365 d": 365, "180 d": 180, "90 d": 90}
TEST_WEEKS = 8
REFERENCE = "hour_of_week"


def point_table(result: pd.DataFrame) -> pd.DataFrame:
    rows = {m: {"pr_auc": pr_auc(result["y"], result[m]), "brier": brier(result["y"], result[m])} for m in BASELINES}
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    alerts, data_end = load_volunteer(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    test_start = data_end.floor("D") - pd.Timedelta(weeks=TEST_WEEKS)
    val_start = test_start - pd.Timedelta(weeks=TEST_WEEKS)
    print(f"data end {data_end:%Y-%m-%d %H:%M} UTC | validation {val_start:%Y-%m-%d}..{test_start:%Y-%m-%d} | "
          f"test {test_start:%Y-%m-%d}..{data_end:%Y-%m-%d}")

    for h in sorted(HORIZONS_H, key=lambda x: x != MAIN_HORIZON_H):
        sample = main_sample(build_frame(alerts, data_end, h))
        runs = {}
        for wname, w in WINDOWS.items():
            runs[wname] = {
                "val": walk_forward(sample, BASELINES, h, val_start, test_start, window_days=w),
                "test": walk_forward(sample, BASELINES, h, test_start, data_end, window_days=w),
            }
        val_brier = {w: brier(r["val"]["y"], r["val"][REFERENCE]) for w, r in runs.items()}
        chosen = min(val_brier, key=val_brier.get)

        print(f"\n===== H = {h} h {'(main)' if h == MAIN_HORIZON_H else ''} =====")
        base_rate = runs[chosen]["test"]["y"].mean()
        print(f"test rows {len(runs[chosen]['test'])}, positive rate {base_rate:.1%}, "
              f"days {runs[chosen]['test']['day'].nunique()}")
        print(f"hour_of_week Brier on validation by window: "
              + ", ".join(f"{w}={v:.4f}" for w, v in val_brier.items()) + f"  -> chosen: {chosen}")

        if h == MAIN_HORIZON_H:
            print("\nInformational: test Brier / PR-AUC of hour_of_week by window (NOT used for choosing)")
            print(pd.DataFrame(
                {w: {"brier": brier(r["test"]["y"], r["test"][REFERENCE]),
                     "pr_auc": pr_auc(r["test"]["y"], r["test"][REFERENCE])} for w, r in runs.items()}
            ).T.round(4).to_string())

        test = runs[chosen]["test"]
        table, diffs = day_block_bootstrap(test, list(BASELINES), REFERENCE, n_boot=args.boot)
        print(f"\nTest, window = {chosen}, day-block bootstrap ({args.boot} draws), 95% interval")
        print(table.round(4).to_string(index=False))
        print(f"\nPaired difference to {REFERENCE} (negative Brier / positive PR-AUC = better than reference)")
        print(diffs.round(4).to_string(index=False))
        if h == MAIN_HORIZON_H:
            print(f"\nCalibration of {REFERENCE} on test (5 quantile bins)")
            print(reliability(test["y"], test[REFERENCE]).to_string())


if __name__ == "__main__":
    main()
