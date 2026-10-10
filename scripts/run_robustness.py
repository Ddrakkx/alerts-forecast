"""Robustness checks of the current conclusions, for the protocol model (best logistic family on validation).

A. Evaluation subsets (same chosen models, same predictions, fewer test rows):
   all rows | without moments whose previous own alert has an invented end (naive) | without the last 24 h of data.
B. Other source: the official file (deduplicated, raions merged into oblast episodes) against the volunteer file,
   both run through the identical protocol on a period that ends before the hole in the official data
   (data end 2026-08-29, so test = 2026-07-04..2026-08-28).
Every line compares with bar B (chosen on validation) AND with the strict reference (best baseline configuration
on the test block, per metric). Bootstrap over ISO weeks.

    python scripts/run_robustness.py [--region ...] [--boot 500]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import REGION, load_official_regions, load_regions  # noqa: E402
from alerts_forecast.experiment import STRICT_BRIER, STRICT_PR, configs, evaluate, feature_index, join_features  # noqa: E402
from alerts_forecast.features import build_features, neighbors_of  # noqa: E402
from alerts_forecast.metrics import day_block_bootstrap, with_blocks  # noqa: E402
from alerts_forecast.target import build_frame, main_sample  # noqa: E402

HORIZONS = (3, 1)


def fmt(v, lo, hi, digits=4) -> str:
    return f"{v:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def summarize(label, res, ev, boot) -> None:
    model = ev.lr_best
    refs = list(dict.fromkeys([ev.bar_b, STRICT_BRIER, STRICT_PR]))
    table, diffs = day_block_bootstrap(with_blocks(res, "week"), [model, *refs], refs, n_boot=boot)
    g = lambda m, k: table[(table.model == m) & (table.metric == k)].iloc[0].value  # noqa: E731
    d = lambda ref, k: diffs[(diffs.model == model) & (diffs.vs == ref) & (diffs.metric == k)].iloc[0]  # noqa: E731
    p_bar, b_bar, p_str, b_str = d(ev.bar_b, "pr_auc"), d(ev.bar_b, "brier"), d(STRICT_PR, "pr_auc"), d(STRICT_BRIER, "brier")
    print(f"  {label:32s} rows={len(res):5d} pos={res['y'].mean():.1%} | model PR-AUC {g(model, 'pr_auc'):.3f}, Brier {g(model, 'brier'):.4f}")
    print(f"      vs bar B {ev.bar_b:20s} dPR {fmt(p_bar['diff'], p_bar.lo, p_bar.hi, 3)} | dBrier {fmt(b_bar['diff'], b_bar.lo, b_bar.hi)}")
    print(f"      vs strict (per metric)          dPR {fmt(p_str['diff'], p_str.lo, p_str.hi, 3)} | dBrier {fmt(b_str['diff'], b_str.lo, b_str.hi)}")


def run_source(name, regions, data_end, h, all_configs, region, test_weeks=8):
    alerts = regions[region]
    nbrs = neighbors_of(region)
    test_start = data_end.floor("D") - pd.Timedelta(weeks=test_weeks)
    val_start = test_start - pd.Timedelta(weeks=test_weeks)
    feats = build_features(regions, region, feature_index(alerts, data_end, [h]), nbrs)
    sample = join_features(main_sample(build_frame(alerts, data_end, h)), feats)
    ev = evaluate(sample, h, val_start, test_start, data_end, all_configs)
    print(f"\n[{name}] H={h}: test {test_start:%Y-%m-%d}..{data_end:%Y-%m-%d}, protocol model {ev.lr_best} "
          f"({ev.chosen[ev.lr_best].split('|', 1)[1]}), bar B {ev.bar_b}, strict {ev.oracle} / {ev.oracle_pr}")
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    all_configs = configs(neighbors_of(args.region))
    print(f"region {args.region} | bootstrap blocks: week")

    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    print("===== A. Evaluation subsets (volunteer, full period) =====")
    for h in HORIZONS:
        ev = run_source("volunteer", regions, data_end, h, all_configs, args.region)
        res = ev.res
        summarize("all rows", res, ev, args.boot)
        summarize("without naive-affected moments", res[~res["prev_naive"]], ev, args.boot)
        summarize("without the last 24 h", res[res.index <= data_end - pd.Timedelta(hours=24 + h)], ev, args.boot)

    print("\n===== B. Volunteer vs official source, identical protocol, data end 2026-08-29 =====")
    end = pd.Timestamp("2026-08-29", tz="UTC").as_unit("ns")
    official = load_official_regions(ROOT / "data" / "raw" / "official_data_en.csv")
    for h in HORIZONS:
        for name, regs in (("volunteer", regions), ("official", official)):
            ev = run_source(name, regs, end, h, all_configs, args.region)
            summarize("all rows", ev.res, ev, args.boot)


if __name__ == "__main__":
    main()
