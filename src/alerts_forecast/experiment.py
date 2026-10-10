"""Shared experiment logic: configurations, validation-based choice, out-of-sample predictions."""
from dataclasses import dataclass

import pandas as pd

from .baselines import constant_rate, hour_of_week_rate, make_smoothed_hour_of_week, recent_activity_rate
from .features import feature_sets
from .metrics import brier, pr_auc
from .models import make_hgb, make_logreg, make_platt
from .target import GRID_STEP, make_grid
from .walkforward import walk_forward

BASE_WINDOWS = {"all": None, "365d": 365, "180d": 180, "90d": 90}
LR_WINDOWS = {"365d": 365, "180d": 180, "90d": 90}  # "all past" lost by a wide margin for every baseline
K_GRID = (25, 100, 400, 1600, 6400)
C_GRID = (0.001, 0.003, 0.01, 0.1)
HGB_GRID = [(d, n) for d in (2, 3) for n in (50, 150)]  # (max_depth, iterations)
HGB_SETS = ("own", "own+nbr+cty")
PRESET_BASELINES = ("constant", "recent_activity", "hour_of_week")
POSTHOC = "hour_of_week_smooth"
RECENT_LEVEL = "recent_level"  # post hoc 2: constant rate of the last W days, refit every day
RL_WINDOWS = {"3d": 3, "7d": 7, "14d": 14, "30d": 30}
POSTHOC_FAMILIES = (POSTHOC, RECENT_LEVEL)
STRICT_BRIER, STRICT_PR = "STRICT_BRIER", "STRICT_PR"  # columns of the strict references in Evaluation.res


def feature_index(alerts, data_end, horizons) -> pd.DatetimeIndex:
    """Grid on which features are built: the one of the SHORTEST horizon, which reaches furthest (end - H)."""
    return make_grid(alerts, data_end, pd.Timedelta(hours=min(horizons)))


def join_features(sample: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Attach features to the sample and refuse to continue if any row lacks them.

    Boosting accepts NaN silently, so a grid mismatch would not crash, it would quietly change the result.
    """
    out = sample.join(feats)
    missing = out[feats.columns].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows without features, first at {out.index[missing][0]}")
    return out


def configs(neighbors, boosting: bool = False) -> list:
    """(family, param, window label, fit_predict) for a region with the given neighbouring oblasts"""
    sets = feature_sets(neighbors)
    out = []
    for w in BASE_WINDOWS:
        out += [("constant", "", w, constant_rate), ("recent_activity", "", w, recent_activity_rate),
                ("hour_of_week", "", w, hour_of_week_rate)]
        out += [(POSTHOC, f"k={k}", w, make_smoothed_hour_of_week(k)) for k in K_GRID]
    for set_name, cols in sets.items():
        for w in LR_WINDOWS:
            for c in C_GRID:
                base = make_logreg(cols, c)
                out.append((f"logreg[{set_name}]", f"C={c}", w, base))
                out.append((f"logreg[{set_name}]+platt", f"C={c}", w, make_platt(base)))
    if boosting:
        for set_name in HGB_SETS:
            for w in LR_WINDOWS:
                out += [(f"hgb[{set_name}]", f"depth={d},iter={n}", w, make_hgb(sets[set_name], d, n)) for d, n in HGB_GRID]
    return out


def run_recent_level(sample, h, start, end) -> pd.DataFrame:
    """Adaptive baseline: base rate of the last W days only, refit every DAY (the other families refit weekly).

    Added after the first results for other oblasts: where the alert rate jumps, static training windows are a weak bar.
    """
    parts = [walk_forward(sample, {f"{RECENT_LEVEL}||{w}": constant_rate}, h, start, end, window_days=d, fold_days=1)
             for w, d in RL_WINDOWS.items()]
    return pd.concat([parts[0][["y", "day"]], *[p.drop(columns=["y", "day"]) for p in parts]], axis=1)


def run_configs(sample, h, start, end, all_configs) -> pd.DataFrame:
    """Out-of-sample predictions of every configuration; one column per configuration label."""
    parts = []
    for w, days in BASE_WINDOWS.items():
        models = {f"{fam}|{par}|{win}": fn for fam, par, win, fn in all_configs if win == w}
        if models:
            parts.append(walk_forward(sample, models, h, start, end, window_days=days))
    parts.append(run_recent_level(sample, h, start, end))
    meta = parts[0][["y", "day"]]
    return pd.concat([meta, *[res.drop(columns=["y", "day"]) for res in parts]], axis=1)


@dataclass
class Evaluation:
    h: int
    val: pd.DataFrame
    test: pd.DataFrame
    val_brier: dict
    chosen: dict      # family -> configuration label chosen on validation
    bar_a: str        # best of the pre-specified baselines
    bar_b: str        # best including the post-hoc baselines (smoothed hour of week, recent level)
    lr_best: str      # logistic family with the best validation Brier: THE model of the protocol
    hgb_best: str     # boosting family with the best validation Brier ("" when boosting was not run)
    oracle: str       # baseline CONFIGURATION with the best Brier on the test block (strict; hindsight, favours baselines)
    oracle_pr: str    # baseline configuration with the best PR-AUC on the test block
    res: pd.DataFrame  # test: y, day, prev_naive, one column per family (chosen configuration), STRICT_BRIER, STRICT_PR


def evaluate(sample, h, val_start, test_start, test_end, all_configs=None) -> Evaluation:
    all_configs = all_configs or configs()
    # purge: validation labels must not look into the test block either (t + H <= test start)
    val = run_configs(sample, h, val_start, test_start - pd.Timedelta(hours=h), all_configs)
    test = run_configs(sample, h, test_start, test_end, all_configs)
    val_brier = {c: brier(val["y"], val[c]) for c in val.columns if "|" in c}
    chosen = {}
    for c, b in val_brier.items():
        fam = c.split("|")[0]
        if fam not in chosen or b < val_brier[chosen[fam]]:
            chosen[fam] = c
    fam_b = lambda f: val_brier[chosen[f]]  # noqa: E731
    bar_a = min((f for f in PRESET_BASELINES if f in chosen), key=fam_b)
    bar_b = min((f for f in (*PRESET_BASELINES, *POSTHOC_FAMILIES) if f in chosen), key=fam_b)
    lr_best = min((f for f in chosen if f.startswith("logreg")), key=fam_b)  # hgb families are compared to it
    res = test[["y", "day"]].copy()
    res["prev_naive"] = sample.loc[res.index, "prev_naive"].to_numpy()
    for fam, c in chosen.items():
        res[fam] = test[c]
    hgbs = [f for f in chosen if f.startswith("hgb")]
    hgb_best = min(hgbs, key=fam_b) if hgbs else ""
    # strict reference: the best of ALL baseline configurations on the test block, separately per metric
    base_cfgs = [c for c in test.columns if "|" in c and c.split("|")[0] in (*PRESET_BASELINES, *POSTHOC_FAMILIES)]
    oracle = min(base_cfgs, key=lambda c: brier(test["y"], test[c]))
    oracle_pr = max(base_cfgs, key=lambda c: pr_auc(test["y"], test[c]))
    res[STRICT_BRIER], res[STRICT_PR] = test[oracle], test[oracle_pr]
    return Evaluation(h, val, test, val_brier, chosen, bar_a, bar_b, lr_best, hgb_best, oracle, oracle_pr, res)
