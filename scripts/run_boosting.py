"""Gradient boosting against the baselines and logistic regression: H=1 and H=3, one look at the test block.

Same protocol as run_experiment.py (validation chooses the configuration of every family by Brier).
Success criterion, written in docs/decisions.md BEFORE the first run: boosting counts as better only if the
paired difference to the best logistic family has a 95% interval entirely above 0 for PR-AUC or entirely
below 0 for Brier, at H=1 or H=3. Otherwise: no gain, and it is reported as such.

    python scripts/run_boosting.py [--boot 1000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import REGION, load_regions  # noqa: E402
from alerts_forecast.experiment import configs, evaluate  # noqa: E402
from alerts_forecast.features import build_features  # noqa: E402
from alerts_forecast.metrics import calibration_slope_intercept, day_block_bootstrap  # noqa: E402
from alerts_forecast.target import build_frame, main_sample  # noqa: E402


def fmt(v, lo, hi, digits=4) -> str:
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    alerts = regions[REGION]
    test_start = data_end.floor("D") - pd.Timedelta(weeks=8)
    val_start = test_start - pd.Timedelta(weeks=8)
    feats = build_features(regions, REGION, build_frame(alerts, data_end, 1).index)
    all_configs = configs(boosting=True)

    for h in (1, 3):
        sample = main_sample(build_frame(alerts, data_end, h)).join(feats)
        ev = evaluate(sample, h, val_start, test_start, data_end, all_configs)
        lr = ev.lr_best
        hgbs = [f for f in ev.chosen if f.startswith("hgb")]
        models = list(dict.fromkeys([ev.bar_b, lr, *hgbs]))
        table, diffs = day_block_bootstrap(ev.res, models, [ev.bar_b, lr], n_boot=args.boot)
        print(f"\n===== H = {h} h | test rows {len(ev.res)}, positive rate {ev.res['y'].mean():.1%} | "
              f"bar B = {ev.bar_b}, best logistic (by validation) = {lr} =====")
        rows = []
        for m in models:
            b = table[(table.model == m) & (table.metric == "brier")].iloc[0]
            p = table[(table.model == m) & (table.metric == "pr_auc")].iloc[0]
            slope, icpt = calibration_slope_intercept(ev.res["y"], ev.res[m])
            rows.append({"model": m, "chosen": ev.chosen[m].split("|", 1)[1], "val_brier": round(ev.val_brier[ev.chosen[m]], 4),
                         "test_brier [95%]": fmt(b.value, b.lo, b.hi), "test_pr_auc [95%]": fmt(p.value, p.lo, p.hi, 3),
                         "cal_slope": round(slope, 2)})
        print(pd.DataFrame(rows).to_string(index=False))
        for ref in dict.fromkeys([lr, ev.bar_b]):
            d = diffs[(diffs.vs == ref) & diffs.model.str.startswith("hgb")].copy()
            d = d.assign(difference=[fmt(r.diff, r.lo, r.hi) for r in d.itertuples()], share_better=d.share_better.round(3))
            print(f"\nPaired difference to {ref}; Brier < 0 and PR-AUC > 0 mean better")
            print(d[["model", "metric", "difference", "share_better"]].to_string(index=False))


if __name__ == "__main__":
    main()
