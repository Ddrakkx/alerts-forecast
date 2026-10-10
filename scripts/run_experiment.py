"""Baselines and logistic regression (with and without Platt recalibration), walk-forward, one test look.

Every model family has a small grid (training window, and k or C). The best configuration of each
family is chosen by Brier on the validation block (8 weeks before the test block). The test block
(last 8 weeks) is then reported for the chosen configurations only, with a day-block bootstrap.

Bars for the models: A = best of the three baselines fixed in advance (constant, recent_activity,
hour_of_week); B = best including the smoothed hour_of_week, which was added post hoc.

    python scripts/run_experiment.py [--boot 1000] [--horizons 3 1 6]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import REGION, load_regions  # noqa: E402
from alerts_forecast.experiment import (  # noqa: E402
    LR_WINDOWS, POSTHOC_FAMILIES, configs, evaluate, feature_index, join_features,
)
from alerts_forecast.features import build_features, feature_sets, neighbors_of  # noqa: E402
from alerts_forecast.metrics import calibration_slope_intercept, day_block_bootstrap, reliability  # noqa: E402
from alerts_forecast.target import MAIN_HORIZON_H, build_frame, main_sample  # noqa: E402
from alerts_forecast.walkforward import split  # noqa: E402

TEST_WEEKS = 8


def fmt(v, lo, hi, digits=4) -> str:
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--horizons", type=float, nargs="+", default=[MAIN_HORIZON_H, 1, 6])
    args = ap.parse_args()

    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    region = args.region
    alerts = regions[region]
    nbrs = neighbors_of(region)
    test_start = data_end.floor("D") - pd.Timedelta(weeks=TEST_WEEKS)
    val_start = test_start - pd.Timedelta(weeks=TEST_WEEKS)
    print(f"region {region} | neighbours {len(nbrs)} | data end {data_end:%Y-%m-%d %H:%M} UTC | validation {val_start:%Y-%m-%d}..{test_start:%Y-%m-%d} | "
          f"test {test_start:%Y-%m-%d}..{data_end:%Y-%m-%d}")

    feats = build_features(regions, region, feature_index(alerts, data_end, args.horizons), nbrs)
    all_configs = configs(nbrs)

    for h in args.horizons:
        t0 = time.time()
        sample = join_features(main_sample(build_frame(alerts, data_end, h)), feats)
        ev = evaluate(sample, h, val_start, test_start, data_end, all_configs)
        res, chosen = ev.res, ev.chosen
        families = list(chosen)
        refs = list(dict.fromkeys([ev.bar_b, ev.bar_a, ev.oracle, ev.oracle_pr, "logreg[own]", "logreg[own+nbr]"]))
        table, diffs = day_block_bootstrap(res, families, refs, n_boot=args.boot)

        print(f"\n===== H = {h:g} h {'(main)' if h == MAIN_HORIZON_H else ''}  [{time.time() - t0:.0f}s] =====")
        print(f"test rows {len(res)}, days {res['day'].nunique()}, positive rate {res['y'].mean():.1%}; "
              f"validation rows {len(ev.val)}, positive rate {ev.val['y'].mean():.1%}")
        print(f"bar A (pre-specified baselines) = {ev.bar_a} | bar B (incl. post-hoc baselines) = {ev.bar_b}")
        print(f"best baseline on the test block by Brier (oracle) = {ev.oracle}")
        print(f"best baseline on the test block by PR-AUC (oracle) = {ev.oracle_pr}")

        rows = []
        for fam in families:
            b = table[(table.model == fam) & (table.metric == "brier")].iloc[0]
            p = table[(table.model == fam) & (table.metric == "pr_auc")].iloc[0]
            slope, icpt = calibration_slope_intercept(res["y"], res[fam])
            rows.append({"family": fam + (" *post-hoc" if fam in POSTHOC_FAMILIES else ""), "chosen": chosen[fam].split("|", 1)[1],
                         "val_brier": round(ev.val_brier[chosen[fam]], 4), "test_brier [95%]": fmt(b.value, b.lo, b.hi),
                         "test_pr_auc [95%]": fmt(p.value, p.lo, p.hi, 3), "cal_slope": round(slope, 2),
                         "cal_icpt": round(icpt, 2)})
        print(pd.DataFrame(rows).to_string(index=False))

        for ref in refs:
            d = diffs[(diffs.vs == ref) & diffs.model.str.startswith("logreg")].copy()
            if d.empty:
                continue
            d = d.assign(difference=[fmt(r.diff, r.lo, r.hi) for r in d.itertuples()], share_better=d.share_better.round(3))
            tag = "".join([" (bar B)" if ref == ev.bar_b else "", " (bar A)" if ref == ev.bar_a else "", " (oracle by Brier)" if ref == ev.oracle else "", " (oracle by PR-AUC)" if ref == ev.oracle_pr else ""])
            print(f"\nPaired difference to {ref}{tag}; Brier < 0 and PR-AUC > 0 mean better")
            print(d[["model", "metric", "difference", "share_better"]].to_string(index=False))

        if h <= 1 or h == MAIN_HORIZON_H:
            for fam in dict.fromkeys([ev.lr_best, ev.lr_best.removesuffix("+platt") + "+platt", ev.bar_b]):
                print(f"\nCalibration on test: {fam} ({chosen[fam].split('|', 1)[1]}), 5 quantile bins")
                print(reliability(res["y"], res[fam]).to_string())
            # which features carry the weight (descriptive: fit on the training window of the last test block)
            fam = ev.lr_best.removesuffix("+platt")
            cols = feature_sets(nbrs)[fam.split("[")[1].rstrip("]")]
            _, par, win = chosen[fam].split("|")
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            last_fold = test_start + pd.Timedelta(days=7 * ((data_end - test_start).days // 7))
            train, _ = split(sample, last_fold, data_end, h, LR_WINDOWS[win])
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=float(par.split("=")[1]), max_iter=2000))
            coef = pd.Series(pipe.fit(train[cols], train["y"].astype(int))[-1].coef_[0], index=cols)
            print(f"\nStandardised coefficients of {fam} (largest |coef|, trained before the last test block)")
            print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
