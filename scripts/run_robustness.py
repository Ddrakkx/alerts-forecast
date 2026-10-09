"""Robustness checks of the main result.

A. Evaluation subsets (same chosen models, same predictions, fewer test rows):
   all rows | without moments whose previous alert has an invented end (naive) | without the last 24 h of data.
B. Other source: the official file (deduplicated, raions merged into oblast episodes) against the volunteer file,
   both run through the identical protocol on a period that ends before the hole in the official data
   (data end 2026-08-29, so test = 2026-07-04..2026-08-28).

    python scripts/run_robustness.py [--boot 500]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import REGION, load_official_regions, load_regions  # noqa: E402
from alerts_forecast.experiment import configs, evaluate  # noqa: E402
from alerts_forecast.features import build_features  # noqa: E402
from alerts_forecast.metrics import day_block_bootstrap  # noqa: E402
from alerts_forecast.target import build_frame, main_sample  # noqa: E402

HORIZONS = (3, 1)
FAMILY = "logreg[own+nbr+cty]"
FAMILY_PLATT = FAMILY + "+platt"


def fmt(v, lo, hi, digits=4) -> str:
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def summarize(label, res, models, ref, boot) -> None:
    table, diffs = day_block_bootstrap(res, models, ref, n_boot=boot)
    g = lambda m, k: table[(table.model == m) & (table.metric == k)].iloc[0]  # noqa: E731
    d = lambda m, k: diffs[(diffs.model == m) & (diffs.metric == k)].iloc[0]  # noqa: E731
    print(f"  {label:34s} rows={len(res):5d} pos={res['y'].mean():.1%} | "
          f"{ref} PR-AUC {g(ref, 'pr_auc').value:.3f}, Brier {g(ref, 'brier').value:.4f}")
    for m in models:
        if m == ref:
            continue
        p, b = d(m, "pr_auc"), d(m, "brier")
        print(f"      {m:24s} PR-AUC {g(m, 'pr_auc').value:.3f}  diff {fmt(p['diff'], p.lo, p.hi, 3)}"
              f" | Brier {g(m, 'brier').value:.4f}  diff {fmt(b['diff'], b.lo, b.hi)}")


def run_source(name, regions, data_end, h, boot, all_configs, test_weeks=8):
    alerts = regions[REGION]
    test_start = data_end.floor("D") - pd.Timedelta(weeks=test_weeks)
    val_start = test_start - pd.Timedelta(weeks=test_weeks)
    feats = build_features(regions, REGION, build_frame(alerts, data_end, 1).index)
    sample = main_sample(build_frame(alerts, data_end, h)).join(feats)
    ev = evaluate(sample, h, val_start, test_start, data_end, all_configs)
    print(f"\n[{name}] H={h}: test {test_start:%Y-%m-%d}..{data_end:%Y-%m-%d}, bar B = {ev.bar_b} "
          f"({ev.chosen[ev.bar_b].split('|', 1)[1]}), logreg chosen = {ev.chosen[FAMILY].split('|', 1)[1]}")
    return ev, data_end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    all_configs = configs()

    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    print("===== A. Evaluation subsets (volunteer, full period) =====")
    for h in HORIZONS:
        ev, _ = run_source("volunteer", regions, data_end, h, args.boot, all_configs)
        res, models = ev.res, [ev.bar_b, FAMILY, FAMILY_PLATT]
        summarize("all rows", res, models, ev.bar_b, args.boot)
        summarize("without naive-affected moments", res[~res["prev_naive"]], models, ev.bar_b, args.boot)
        cut = data_end - pd.Timedelta(hours=24 + h)
        summarize("without the last 24 h", res[res.index <= cut], models, ev.bar_b, args.boot)

    print("\n===== B. Volunteer vs official source, identical protocol, data end 2026-08-29 =====")
    end = pd.Timestamp("2026-08-29", tz="UTC").as_unit("ns")
    official = load_official_regions(ROOT / "data" / "raw" / "official_data_en.csv")
    for h in HORIZONS:
        for name, regs in (("volunteer", regions), ("official", official)):
            ev, _ = run_source(name, regs, end, h, args.boot, all_configs)
            summarize("all rows", ev.res, [ev.bar_b, FAMILY, FAMILY_PLATT], ev.bar_b, args.boot)


if __name__ == "__main__":
    main()
