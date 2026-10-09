"""Baselines and logistic regression with three feature sets, walk-forward, one test look.

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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.baselines import (  # noqa: E402
    constant_rate, hour_of_week_rate, make_smoothed_hour_of_week, recent_activity_rate,
)
from alerts_forecast.data import REGION, load_regions  # noqa: E402
from alerts_forecast.features import FEATURE_SETS, build_features  # noqa: E402
from alerts_forecast.metrics import brier, day_block_bootstrap, pr_auc, reliability  # noqa: E402
from alerts_forecast.models import make_logreg  # noqa: E402
from alerts_forecast.target import MAIN_HORIZON_H, build_frame, main_sample  # noqa: E402
from alerts_forecast.walkforward import split, walk_forward  # noqa: E402

TEST_WEEKS = 8
BASE_WINDOWS = {"all": None, "365d": 365, "180d": 180, "90d": 90}
LR_WINDOWS = {"365d": 365, "180d": 180, "90d": 90}  # "all past" lost by a wide margin for every baseline
K_GRID = (25, 100, 400, 1600)
C_GRID = (0.01, 0.1, 1.0)
PRESET_BASELINES = ("constant", "recent_activity", "hour_of_week")
POSTHOC = "hour_of_week_smooth"


def configs() -> list:
    """(family, param, window label, fit_predict)"""
    out = []
    for w in BASE_WINDOWS:
        out += [("constant", "", w, constant_rate), ("recent_activity", "", w, recent_activity_rate),
                ("hour_of_week", "", w, hour_of_week_rate)]
        out += [(POSTHOC, f"k={k}", w, make_smoothed_hour_of_week(k)) for k in K_GRID]
    for set_name, cols in FEATURE_SETS.items():
        for w in LR_WINDOWS:
            out += [(f"logreg[{set_name}]", f"C={c}", w, make_logreg(cols, c)) for c in C_GRID]
    return out


def run_configs(sample, h, start, end, all_configs) -> pd.DataFrame:
    """Out-of-sample predictions of every configuration; one column per configuration label."""
    parts = {}
    for w, days in BASE_WINDOWS.items():
        models = {f"{fam}|{par}|{win}": fn for fam, par, win, fn in all_configs if win == w}
        if models:
            res = walk_forward(sample, models, h, start, end, window_days=days)
            parts[w] = res
    first = next(iter(parts.values()))
    out = first[["y", "day"]].copy()
    for res in parts.values():
        for col in res.columns.difference(["y", "day"]):
            out[col] = res[col]
    return out


def fmt(v, lo, hi, digits=4) -> str:
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--horizons", type=int, nargs="+", default=[MAIN_HORIZON_H, 1, 6])
    args = ap.parse_args()

    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    alerts = regions[REGION]
    test_start = data_end.floor("D") - pd.Timedelta(weeks=TEST_WEEKS)
    val_start = test_start - pd.Timedelta(weeks=TEST_WEEKS)
    print(f"data end {data_end:%Y-%m-%d %H:%M} UTC | validation {val_start:%Y-%m-%d}..{test_start:%Y-%m-%d} | "
          f"test {test_start:%Y-%m-%d}..{data_end:%Y-%m-%d}")

    t0 = time.time()
    longest = build_frame(alerts, data_end, 1)  # the H=1 grid is the longest one; others are subsets
    feats = build_features(regions, REGION, longest.index)
    print(f"features: {feats.shape[1]} columns x {len(feats)} moments in {time.time() - t0:.0f}s")
    all_configs = configs()

    for h in args.horizons:
        t0 = time.time()
        sample = main_sample(build_frame(alerts, data_end, h)).join(feats)
        val = run_configs(sample, h, val_start, test_start, all_configs)
        test = run_configs(sample, h, test_start, data_end, all_configs)

        # choose the best configuration of every family on validation
        val_brier = {c: brier(val["y"], val[c]) for c in val.columns if "|" in c}
        chosen = {}
        for c, b in val_brier.items():
            fam = c.split("|")[0]
            if fam not in chosen or b < val_brier[chosen[fam]]:
                chosen[fam] = c
        bar_a = min(PRESET_BASELINES, key=lambda f: val_brier[chosen[f]])
        bar_b = min((*PRESET_BASELINES, POSTHOC), key=lambda f: val_brier[chosen[f]])
        lr_best = min((f for f in chosen if f.startswith("logreg")), key=lambda f: val_brier[chosen[f]])

        res = test[["y", "day"]].copy()
        for fam, c in chosen.items():
            res[fam] = test[c]
        families = list(chosen)
        refs = list(dict.fromkeys([bar_b, bar_a]))
        refs_models = refs + ["logreg[own]", "logreg[own+nbr]"]
        table, diffs = day_block_bootstrap(res, families, list(dict.fromkeys(refs_models)), n_boot=args.boot)

        print(f"\n===== H = {h} h {'(main)' if h == MAIN_HORIZON_H else ''}  [{time.time() - t0:.0f}s] =====")
        print(f"test rows {len(res)}, days {res['day'].nunique()}, positive rate {res['y'].mean():.1%}; "
              f"validation rows {len(val)}, positive rate {val['y'].mean():.1%}")
        print(f"bar A (pre-specified baselines) = {bar_a} | bar B (incl. post-hoc smoothed) = {bar_b}")

        rows = []
        for fam in families:
            b = table[(table.model == fam) & (table.metric == "brier")].iloc[0]
            p = table[(table.model == fam) & (table.metric == "pr_auc")].iloc[0]
            rows.append({"family": fam + (" *post-hoc" if fam == POSTHOC else ""), "chosen": chosen[fam].split("|", 1)[1],
                         "val_brier": round(val_brier[chosen[fam]], 4), "test_brier [95%]": fmt(b.value, b.lo, b.hi),
                         "test_pr_auc [95%]": fmt(p.value, p.lo, p.hi, 3)})
        print(pd.DataFrame(rows).to_string(index=False))

        for ref in dict.fromkeys(refs_models):
            d = diffs[(diffs.vs == ref) & diffs.model.str.startswith("logreg")].copy()
            if ref.startswith("logreg"):
                d = d[d.model != ref]
            if d.empty:
                continue
            label = {bar_b: "bar B", bar_a: "bar A"}.get(ref, ref)
            if ref == bar_b == bar_a:
                label = "bar A = bar B"
            d = d.assign(difference=[fmt(r.diff, r.lo, r.hi) for r in d.itertuples()],
                         share_better=d.share_better.round(3))
            print(f"\nPaired difference to {ref} ({label}); Brier < 0 and PR-AUC > 0 mean better")
            print(d[["model", "metric", "difference", "share_better"]].to_string(index=False))

        if h in (MAIN_HORIZON_H, 1):
            for fam in (lr_best, bar_b):
                print(f"\nCalibration on test: {fam} ({chosen[fam].split('|', 1)[1]}), 5 quantile bins")
                print(reliability(res["y"], res[fam]).to_string())
            # which features carry the weight (descriptive: fit on the training window of the last test block)
            cols = FEATURE_SETS[lr_best.split("[")[1].rstrip("]")]
            c_val = float(chosen[lr_best].split("|")[1].split("=")[1])
            window = LR_WINDOWS[chosen[lr_best].split("|")[2]]
            last_fold = (test_start + pd.Timedelta(days=7 * ((data_end - test_start).days // 7)))
            train, _ = split(sample, last_fold, data_end, h, window)
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c_val, max_iter=2000)).fit(train[cols], train["y"].astype(int))
            coef = pd.Series(pipe[-1].coef_[0], index=cols)
            print(f"\nStandardised coefficients of {lr_best} (largest |coef|, trained before the last test block)")
            print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
