"""Holdout test on a NEW snapshot, exactly as pre-registered in docs/decisions.md (decision 9).

1. Compare the old and the new snapshot for alerts that started before the old end (added / removed / changed per oblast).
2. For every oblast and horizon: validation = the 8 weeks before the holdout start (purged by H), configurations chosen by
   validation Brier, protocol model (lr_best) vs bar B; hindsight bound vs the best baseline configuration on the holdout.
   Paired bootstrap over 6-hour blocks of the Kyiv clock, 1000 draws, seed 0. A difference counts only if its 95%
   interval excludes zero; otherwise "inconclusive". A cell with fewer than 10 positive or 10 negative moments is "not evaluable".

    python scripts/run_holdout.py --new data/holdout/volunteer_data_en.csv          # the real run (Monday)
    python scripts/run_holdout.py --simulate-days 3                                  # dry run on the current snapshot (code check only)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import load_regions  # noqa: E402
from alerts_forecast.experiment import STRICT_BRIER, STRICT_PR, configs, evaluate, feature_index, join_features  # noqa: E402
from alerts_forecast.features import build_features, neighbors_of  # noqa: E402
from alerts_forecast.metrics import day_block_bootstrap, with_blocks  # noqa: E402
from alerts_forecast.target import build_frame, main_sample  # noqa: E402

OLD_FILE = ROOT / "data" / "raw" / "volunteer_data_en.csv"
OLD_END = pd.Timestamp("2026-10-09 05:12:41", tz="UTC")  # end of the snapshot used for all earlier results
CELLS = [  # (oblast, horizon in hours, role)
    ("Poltavska oblast", 3.0, "PRIMARY (pre-registered)"),
    ("Kyivska oblast", 0.25, "secondary (pre-registered)"), ("Kyivska oblast", 0.5, "secondary (pre-registered)"),
    ("Kyivska oblast", 1.0, "secondary (pre-registered)"),
    ("Kharkivska oblast", 0.25, "secondary (pre-registered)"), ("Kharkivska oblast", 0.5, "secondary (pre-registered)"),
    ("Kharkivska oblast", 1.0, "secondary (pre-registered)"),
    ("Lvivska oblast", 0.25, "reported with the data warning"), ("Lvivska oblast", 0.5, "reported with the data warning"),
    ("Lvivska oblast", 1.0, "reported with the data warning"), ("Lvivska oblast", 3.0, "reported with the data warning"),
]
MIN_CLASS = 10


def fmt(d, metric: str) -> str:
    """For PR-AUC higher is better, for Brier lower is better."""
    v, lo, hi = d
    up, down = ("better", "worse") if metric == "pr_auc" else ("worse", "better")
    return f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}] -> " + (up if lo > 0 else down if hi < 0 else "inconclusive")


def compare_snapshots(old_path, new_path, old_end) -> None:
    """Records that started before the old end: added, removed, changed (same oblast and start, other end or flag)."""
    cols = ["region", "started_at", "finished_at", "naive"]
    old, new = (pd.read_csv(p)[cols] for p in (old_path, new_path))
    for df in (old, new):
        df["t"] = pd.to_datetime(df["started_at"], utc=True)
    old, new = old[old["t"] <= old_end], new[new["t"] <= old_end]
    key = ["region", "started_at"]
    m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    added = m[m["_merge"] == "right_only"]
    removed = m[m["_merge"] == "left_only"]
    both = m[m["_merge"] == "both"]
    changed = both[(both["finished_at_old"] != both["finished_at_new"]) | (both["naive_old"] != both["naive_new"])]
    print(f"Snapshot comparison for alerts that started before {old_end:%Y-%m-%d %H:%M:%S} UTC: "
          f"added {len(added)}, removed {len(removed)}, changed {len(changed)}")
    for name, part in (("added", added), ("removed", removed), ("changed", changed)):
        if len(part):
            print(f"  {name}: " + ", ".join(f"{r} {n}" for r, n in part["region"].value_counts().items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, help="the new snapshot (volunteer file)")
    ap.add_argument("--simulate-days", type=float, help="dry run: treat the last N days of the current snapshot as the holdout")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    if args.simulate_days:
        new_path = OLD_FILE
        _, current_end = load_regions(OLD_FILE)
        old_end = current_end - pd.Timedelta(days=args.simulate_days)
        print(f"DRY RUN on already seen data (code check only): holdout = last {args.simulate_days:g} days of the current snapshot")
    else:
        new_path, old_end = args.new, OLD_END
        compare_snapshots(OLD_FILE, new_path, old_end)

    regions, new_end = load_regions(new_path)
    holdout_start = old_end
    val_start = holdout_start - pd.Timedelta(weeks=8)
    print(f"holdout {holdout_start:%Y-%m-%d %H:%M} .. {new_end:%Y-%m-%d %H:%M} UTC "
          f"({(new_end - holdout_start) / pd.Timedelta(hours=1):.1f} h); validation {val_start:%Y-%m-%d %H:%M} .. holdout start - H; "
          f"bootstrap: 6-hour blocks, {args.boot} draws, seed 0")

    for region in dict.fromkeys(c[0] for c in CELLS):
        nbrs = neighbors_of(region)
        horizons = [h for r, h, _ in CELLS if r == region]
        alerts = regions[region]
        feats = build_features(regions, region, feature_index(alerts, new_end, horizons), nbrs)
        cfg = configs(nbrs)
        for r, h, role in CELLS:
            if r != region:
                continue
            sample = join_features(main_sample(build_frame(alerts, new_end, h)), feats)
            hold = sample[sample.index > holdout_start]
            pos, neg = int(hold["y"].sum()), int((~hold["y"].astype(bool)).sum())
            head = f"\n[{region}, H = {h:g} h, {role}] holdout moments {len(hold)}, positive {pos}, negative {neg}"
            if pos < MIN_CLASS or neg < MIN_CLASS:
                print(head + " -> NOT EVALUABLE (fewer than 10 of a class)")
                continue
            ev = evaluate(sample, h, val_start, holdout_start, new_end, cfg)
            res = with_blocks(ev.res, "6h")
            model, refs = ev.lr_best, list(dict.fromkeys([ev.bar_b, STRICT_BRIER, STRICT_PR]))
            table, diffs = day_block_bootstrap(res, [model, *refs], refs, n_boot=args.boot, seed=0)
            d = lambda ref, k: tuple(diffs[(diffs.model == model) & (diffs.vs == ref) & (diffs.metric == k)].iloc[0][["diff", "lo", "hi"]])  # noqa: E731
            v = lambda m, k: table[(table.model == m) & (table.metric == k)].iloc[0].value  # noqa: E731
            print(head + f", blocks {res['day'].nunique()}")
            print(f"  protocol model {model} ({ev.chosen[model].split('|', 1)[1]}): PR-AUC {v(model, 'pr_auc'):.3f}, Brier {v(model, 'brier'):.4f}")
            print(f"  bar B {ev.bar_b} ({ev.chosen[ev.bar_b].split('|', 1)[1]}): PR-AUC {v(ev.bar_b, 'pr_auc'):.3f}, Brier {v(ev.bar_b, 'brier'):.4f}")
            print(f"  vs bar B (the pre-registered comparison): dPR-AUC {fmt(d(ev.bar_b, 'pr_auc'), 'pr_auc')} | dBrier {fmt(d(ev.bar_b, 'brier'), 'brier')}")
            print(f"  hindsight bound vs best baseline configuration on the holdout ({ev.oracle_pr} / {ev.oracle}): "
                  f"dPR-AUC {fmt(d(STRICT_PR, 'pr_auc'), 'pr_auc')} | dBrier {fmt(d(STRICT_BRIER, 'brier'), 'brier')}")


if __name__ == "__main__":
    main()
