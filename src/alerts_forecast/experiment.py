"""Shared experiment logic: configurations, validation-based choice, out-of-sample predictions."""
from dataclasses import dataclass

import pandas as pd

from .baselines import constant_rate, hour_of_week_rate, make_smoothed_hour_of_week, recent_activity_rate
from .features import FEATURE_SETS
from .metrics import brier
from .models import make_logreg, make_platt
from .walkforward import walk_forward

BASE_WINDOWS = {"all": None, "365d": 365, "180d": 180, "90d": 90}
LR_WINDOWS = {"365d": 365, "180d": 180, "90d": 90}  # "all past" lost by a wide margin for every baseline
K_GRID = (25, 100, 400, 1600, 6400)
C_GRID = (0.001, 0.003, 0.01, 0.1)
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
            for c in C_GRID:
                base = make_logreg(cols, c)
                out.append((f"logreg[{set_name}]", f"C={c}", w, base))
                out.append((f"logreg[{set_name}]+platt", f"C={c}", w, make_platt(base)))
    return out


def run_configs(sample, h, start, end, all_configs) -> pd.DataFrame:
    """Out-of-sample predictions of every configuration; one column per configuration label."""
    parts = []
    for w, days in BASE_WINDOWS.items():
        models = {f"{fam}|{par}|{win}": fn for fam, par, win, fn in all_configs if win == w}
        if models:
            parts.append(walk_forward(sample, models, h, start, end, window_days=days))
    out = parts[0][["y", "day"]].copy()
    for res in parts:
        for col in res.columns.difference(["y", "day"]):
            out[col] = res[col]
    return out


@dataclass
class Evaluation:
    h: int
    val: pd.DataFrame
    test: pd.DataFrame
    val_brier: dict
    chosen: dict      # family -> configuration label chosen on validation
    bar_a: str        # best of the pre-specified baselines
    bar_b: str        # best including the post-hoc smoothed one
    lr_best: str      # logistic family with the best validation Brier
    res: pd.DataFrame  # test: y, day, prev_naive and one column per family (chosen configuration)


def evaluate(sample, h, val_start, test_start, test_end, all_configs=None) -> Evaluation:
    all_configs = all_configs or configs()
    val = run_configs(sample, h, val_start, test_start, all_configs)
    test = run_configs(sample, h, test_start, test_end, all_configs)
    val_brier = {c: brier(val["y"], val[c]) for c in val.columns if "|" in c}
    chosen = {}
    for c, b in val_brier.items():
        fam = c.split("|")[0]
        if fam not in chosen or b < val_brier[chosen[fam]]:
            chosen[fam] = c
    fam_b = lambda f: val_brier[chosen[f]]  # noqa: E731
    bar_a = min(PRESET_BASELINES, key=fam_b)
    bar_b = min((*PRESET_BASELINES, POSTHOC), key=fam_b)
    lr_best = min((f for f in chosen if f.startswith("logreg")), key=fam_b)
    res = test[["y", "day"]].copy()
    res["prev_naive"] = sample.loc[res.index, "prev_naive"].to_numpy()
    for fam, c in chosen.items():
        res[fam] = test[c]
    return Evaluation(h, val, test, val_brier, chosen, bar_a, bar_b, lr_best, res)
