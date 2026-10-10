"""Gradient boosting against the baselines and logistic regression, one oblast.

Same protocol as run_experiment.py (validation chooses the configuration of every family by Brier; the protocol
boosting model is the boosting family with the best validation Brier). Bootstrap over ISO weeks by default.
Success criterion of decision 6, written before the first run: boosting counts as better only if the paired difference
to the protocol logistic model has a 95% interval entirely above 0 for PR-AUC or entirely below 0 for Brier.

    python scripts/run_boosting.py [--region ...] [--horizons 1 3] [--block week] [--boot 1000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import REGION, load_regions  # noqa: E402
from alerts_forecast.experiment import STRICT_BRIER, STRICT_PR, configs, evaluate, feature_index, join_features  # noqa: E402
from alerts_forecast.features import build_features, neighbors_of  # noqa: E402
from alerts_forecast.metrics import calibration_slope_intercept, day_block_bootstrap, with_blocks  # noqa: E402
from alerts_forecast.target import build_frame, main_sample  # noqa: E402


def fmt(v, lo, hi, digits=4) -> str:
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--horizons", type=float, nargs="+", default=[1, 3])
    ap.add_argument("--block", default="week", choices=["week", "day", "6h"])
    args = ap.parse_args()
    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    region = args.region
    alerts = regions[region]
    nbrs = neighbors_of(region)
    print(f"region {region} | neighbours {len(nbrs)} | bootstrap blocks: {args.block}")
    test_start = data_end.floor("D") - pd.Timedelta(weeks=8)
    val_start = test_start - pd.Timedelta(weeks=8)
    feats = build_features(regions, region, feature_index(alerts, data_end, args.horizons), nbrs)
    all_configs = configs(nbrs, boosting=True)

    for h in args.horizons:
        sample = join_features(main_sample(build_frame(alerts, data_end, h)), feats)
        ev = evaluate(sample, h, val_start, test_start, data_end, all_configs)
        lr, hb = ev.lr_best, ev.hgb_best
        hgbs = [f for f in ev.chosen if f.startswith("hgb")]
        chosen = {**ev.chosen, STRICT_BRIER: ev.oracle, STRICT_PR: ev.oracle_pr}
        models = list(dict.fromkeys([ev.bar_b, STRICT_BRIER, STRICT_PR, lr, *hgbs]))
        res = with_blocks(ev.res, args.block)
        table, diffs = day_block_bootstrap(res, models, [ev.bar_b, lr, STRICT_BRIER, STRICT_PR], n_boot=args.boot)
        print(f"\n===== H = {h:g} h | test rows {len(res)}, positive rate {res['y'].mean():.1%} | "
              f"bar B = {ev.bar_b}, protocol logistic = {lr}, protocol boosting = {hb}, "
              f"strict by Brier = {ev.oracle}, strict by PR-AUC = {ev.oracle_pr} =====")
        rows = []
        for m in models:
            b = table[(table.model == m) & (table.metric == "brier")].iloc[0]
            p = table[(table.model == m) & (table.metric == "pr_auc")].iloc[0]
            slope, icpt = calibration_slope_intercept(res["y"], res[m])
            label = chosen[m] if m in (STRICT_BRIER, STRICT_PR) else chosen[m].split("|", 1)[1]
            rows.append({"model": m, "chosen": label, "val_brier": round(ev.val_brier[chosen[m]], 4),
                         "test_brier [95%]": fmt(b.value, b.lo, b.hi), "test_pr_auc [95%]": fmt(p.value, p.lo, p.hi, 3),
                         "cal_slope": round(slope, 2)})
        print(pd.DataFrame(rows).to_string(index=False))
        for ref in dict.fromkeys([lr, ev.bar_b, STRICT_BRIER, STRICT_PR]):
            d = diffs[(diffs.vs == ref) & diffs.model.str.startswith("hgb")].copy()
            d = d.assign(difference=[fmt(r.diff, r.lo, r.hi) for r in d.itertuples()], share_better=d.share_better.round(3))
            print(f"\nPaired difference to {ref}; Brier < 0 and PR-AUC > 0 mean better")
            print(d[["model", "metric", "difference", "share_better"]].to_string(index=False))

        # POST HOC, added after the first run showed that the reference of the criterion was a weak logistic variant
        top = "hgb[own+nbr+cty]"
        others = [f for f in ev.chosen if f != top]
        _, d2 = day_block_bootstrap(res, [top, *others], others, n_boot=args.boot)
        d2 = d2[d2.model == top].copy()
        d2 = d2.assign(difference=[fmt(r.diff, r.lo, r.hi) for r in d2.itertuples()], share_better=d2.share_better.round(3))
        print(f"\nPOST HOC (not part of the criterion): {top} against every other chosen family")
        print(d2[["vs", "metric", "difference", "share_better"]].sort_values(["metric", "vs"]).to_string(index=False))


if __name__ == "__main__":
    main()
