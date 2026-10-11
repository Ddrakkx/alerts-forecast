"""Amended holdout analysis (decision 15), registered before the new snapshot was downloaded.

What changes against the frozen scripts/run_holdout.py (which is still run and reported unchanged):
- the new snapshot is cut at E = its end - 6 h, as known at E: alerts still running at the snapshot end are missing from the file,
  so the labels of the last hours would be falsely 0; holdout moments are old end < t <= E - H;
- the snapshot comparison splits added records into "running at the old end" (structurally missing) and true late additions,
  and compares starts as times, not as strings;
- a cell is evaluable with at least 5 own alert starts in (old end, E] and 10 negative moments (events, not moments);
- no verdicts: the differences come with 6-hour-block intervals ("these days only") next to the spread of the same difference
  over 3-day windows of the old test block (results/holdout/reference_spread.csv, computed from seen data with --reference);
- an overlap inside one oblast is merged with a warning instead of stopping the run.
Cells, validation, purge, selection (lr_best vs bar B) and the strict hindsight bound are those of decision 9.

    python analysis/holdout_amended.py --reference                                  # spread over 3-day windows, old data (Sunday)
    python analysis/holdout_amended.py --new data/holdout/volunteer_data_en.csv     # the real run, writes results/holdout/amended.txt
    python analysis/holdout_amended.py --simulate-days 3                            # dry run on seen data (code check only)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from alerts_forecast.data import EXCLUDED_REGIONS, REQUIRED_COLUMNS, _read, validate_alerts  # noqa: E402
from alerts_forecast.experiment import STRICT_BRIER, STRICT_PR, configs, evaluate, feature_index, join_features  # noqa: E402
from alerts_forecast.features import build_features, neighbors_of  # noqa: E402
from alerts_forecast.leakage import truncate_regions  # noqa: E402
from alerts_forecast.metrics import brier, day_block_bootstrap, pr_auc, with_blocks  # noqa: E402
from alerts_forecast.target import build_frame, main_sample  # noqa: E402
from run_holdout import CELLS, OLD_END, OLD_FILE  # noqa: E402  (the frozen cell list, so both runners judge the same cells)

TRIM = pd.Timedelta(hours=6)
MIN_STARTS, MIN_NEG = 5, 10
WINDOW = pd.Timedelta(days=3)
TEST_WEEKS = 8
OUT_DIR = ROOT / "results" / "holdout"
REFERENCE = OUT_DIR / "reference_spread.csv"


# ---------------------------------------------------------------- data
def load_robust(path) -> tuple[dict, pd.Timestamp, list]:
    """Like data.load_regions, but an overlap inside one oblast is merged (naive if any part was) and reported, not fatal."""
    raw, data_end = _read(path)
    regions, warnings = {}, []
    for r in sorted(raw["region"].unique()):
        if r in EXCLUDED_REGIONS:
            continue
        a = raw.loc[raw["region"] == r, ["started_at", "finished_at", "naive"]].sort_values("started_at").reset_index(drop=True)
        group = (a["started_at"] > a["finished_at"].cummax().shift()).cumsum()
        if group.nunique() < len(a):
            warnings.append(f"{r}: {len(a) - group.nunique()} overlapping records merged")
            a = a.groupby(group).agg(started_at=("started_at", "min"), finished_at=("finished_at", "max"), naive=("naive", "any"))
            a = a.reset_index(drop=True)
        validate_alerts(a)
        regions[r] = a
    return regions, data_end, warnings


def cut(regions: dict, data_end: pd.Timestamp) -> tuple[dict, pd.Timestamp]:
    """The data as known at E = end - TRIM: later alerts absent, ends after E unknown."""
    e = data_end - TRIM
    return truncate_regions(regions, e), e


def own_starts(alerts: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(((alerts["started_at"] > a) & (alerts["started_at"] <= b)).sum())


# ---------------------------------------------------------------- snapshot comparison
def compare_snapshots(old: pd.DataFrame, new: pd.DataFrame, old_end: pd.Timestamp) -> dict:
    """Records that started before the old end. Starts are compared as times. Added ones are split into those still running
    at the old end (a snapshot never holds them, so they are not late) and those that had finished: true late additions."""
    def prep(df):
        d = df[list(REQUIRED_COLUMNS)].copy()
        for c in ("started_at", "finished_at"):
            d[c] = pd.to_datetime(d[c], utc=True, format="ISO8601")
        return d[d["started_at"] <= old_end]
    m = prep(old).merge(prep(new), on=["region", "started_at"], how="outer", suffixes=("_old", "_new"), indicator=True)
    added = m[m["_merge"] == "right_only"]
    both = m[m["_merge"] == "both"]
    return {
        "running": added[added["finished_at_new"] > old_end],
        "late": added[added["finished_at_new"] <= old_end],
        "removed": m[m["_merge"] == "left_only"],
        "changed": both[(both["finished_at_old"] != both["finished_at_new"]) | (both["naive_old"] != both["naive_new"])],
    }


def comparison_lines(c: dict, old_end: pd.Timestamp) -> list:
    lines = [f"Snapshot comparison, alerts that started before {old_end:%Y-%m-%d %H:%M:%S} UTC: "
             f"running at the old end {len(c['running'])}, true late additions {len(c['late'])}, removed {len(c['removed'])}, changed {len(c['changed'])}"]
    for name in ("running", "removed", "changed"):
        if len(c[name]):
            lines.append(f"  {name}: " + ", ".join(f"{r} {n}" for r, n in c[name]["region"].value_counts().items()))
    for _, row in c["late"].iterrows():
        lines.append(f"  late: {row['region']} {row['started_at']:%Y-%m-%d %H:%M} .. {row['finished_at_new']:%Y-%m-%d %H:%M} naive={row['naive_new']}")
    return lines


# ---------------------------------------------------------------- evaluation
def region_features(regions, region, end):
    horizons = [h for r, h, _ in CELLS if r == region]
    return horizons, build_features(regions, region, feature_index(regions[region], end, horizons), neighbors_of(region))


def reference_spread(regions: dict, old_end: pd.Timestamp) -> pd.DataFrame:
    """Old test block (8 weeks, cut at old end - TRIM like the holdout), fixed protocol models, 3-day windows:
    how much does the difference model - bar B move between windows of the holdout's length?"""
    regions, e = cut(regions, old_end)
    test_start = old_end.floor("D") - pd.Timedelta(weeks=TEST_WEEKS)
    val_start = test_start - pd.Timedelta(weeks=TEST_WEEKS)
    rows = []
    for region in dict.fromkeys(c[0] for c in CELLS):
        horizons, feats = region_features(regions, region, e)
        for h in horizons:
            sample = join_features(main_sample(build_frame(regions[region], e, h)), feats)
            ev = evaluate(sample, h, val_start, test_start, e, configs(neighbors_of(region)))
            res, model, ref = ev.res, ev.lr_best, ev.bar_b
            d = {"pr_auc": [], "brier": []}
            for a in pd.date_range(test_start, e, freq=WINDOW)[:-1]:
                w = res[(res.index > a) & (res.index <= a + WINDOW)]
                if own_starts(regions[region], a, a + WINDOW) < MIN_STARTS or (~w["y"].astype(bool)).sum() < MIN_NEG or w["y"].nunique() < 2:
                    continue
                d["pr_auc"].append(pr_auc(w["y"], w[model]) - pr_auc(w["y"], w[ref]))
                d["brier"].append(brier(w["y"], w[model]) - brier(w["y"], w[ref]))
            row = {"region": region, "h": h, "model": model, "bar_b": ref, "windows": len(d["brier"]),
                   "full_pr_auc": pr_auc(res["y"], res[model]) - pr_auc(res["y"], res[ref]),
                   "full_brier": brier(res["y"], res[model]) - brier(res["y"], res[ref])}
            for k, v in d.items():
                ok = len(v) >= 2
                row[f"sd_{k}"] = np.std(v, ddof=1) if ok else np.nan
                row[f"p05_{k}"], row[f"p95_{k}"] = (np.percentile(v, 5), np.percentile(v, 95)) if ok else (np.nan, np.nan)
            rows.append(row)
            print(f"  {region} H={h:g}: {len(d['brier'])} windows", flush=True)
    return pd.DataFrame(rows)


def holdout(regions: dict, data_end: pd.Timestamp, old_end: pd.Timestamp, boot: int, ref: pd.DataFrame) -> list:
    regions, e = cut(regions, data_end)
    val_start = old_end - pd.Timedelta(weeks=TEST_WEEKS)
    lines = [f"holdout {old_end:%Y-%m-%d %H:%M} .. E - H, E = end - 6 h = {e:%Y-%m-%d %H:%M} UTC (E - old end = {(e - old_end) / pd.Timedelta(hours=1):.1f} h); "
             f"validation {val_start:%Y-%m-%d %H:%M} .. holdout start - H; intervals: 6-hour blocks, {boot} draws, seed 0, these days only. No verdicts."]
    for region in dict.fromkeys(c[0] for c in CELLS):
        _, feats = region_features(regions, region, e)
        n_starts = own_starts(regions[region], old_end, e)
        for r, h, role in CELLS:
            if r != region:
                continue
            sample = join_features(main_sample(build_frame(regions[region], e, h)), feats)
            hold = sample[sample.index > old_end]
            pos, neg = int(hold["y"].sum()), int((~hold["y"].astype(bool)).sum())
            head = f"\n[{region}, H = {h:g} h, {role}] holdout moments {len(hold)}, own alert starts {n_starts}, positive {pos}, negative {neg}"
            if n_starts < MIN_STARTS or neg < MIN_NEG:
                lines.append(head + f" -> NOT EVALUABLE (needs {MIN_STARTS} own starts and {MIN_NEG} negative moments)")
                continue
            ev = evaluate(sample, h, val_start, old_end, e, configs(neighbors_of(region)))
            res = with_blocks(ev.res, "6h")
            model, refs = ev.lr_best, list(dict.fromkeys([ev.bar_b, STRICT_BRIER, STRICT_PR]))
            table, diffs = day_block_bootstrap(res, [model, *refs], refs, n_boot=boot, seed=0)
            d = lambda vs, k: diffs[(diffs.model == model) & (diffs.vs == vs) & (diffs.metric == k)].iloc[0]  # noqa: E731
            v = lambda m, k: table[(table.model == m) & (table.metric == k)].iloc[0].value  # noqa: E731
            iv = lambda x: f"{x['diff']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"  # noqa: E731
            lines += [head + f", 6-hour blocks {res['day'].nunique()}",
                      f"  protocol model {model} ({ev.chosen[model].split('|', 1)[1]}): PR-AUC {v(model, 'pr_auc'):.3f}, Brier {v(model, 'brier'):.4f}",
                      f"  bar B {ev.bar_b} ({ev.chosen[ev.bar_b].split('|', 1)[1]}): PR-AUC {v(ev.bar_b, 'pr_auc'):.3f}, Brier {v(ev.bar_b, 'brier'):.4f}",
                      f"  vs bar B, these days only: dPR-AUC {iv(d(ev.bar_b, 'pr_auc'))} | dBrier {iv(d(ev.bar_b, 'brier'))}"]
            rr = ref[(ref.region == region) & (ref.h == h) & (ref.windows >= 2)]
            if len(rr):
                rr = rr.iloc[0]
                pos_of = lambda k: "inside" if rr[f"p05_{k}"] <= d(ev.bar_b, k)["diff"] <= rr[f"p95_{k}"] else "OUTSIDE"  # noqa: E731
                lines.append(f"  old test block, {int(rr.windows)} windows of 3 days ({rr.model} vs {rr.bar_b}): dPR-AUC 8 weeks {rr.full_pr_auc:+.4f}, "
                             f"middle 90% of windows [{rr.p05_pr_auc:+.4f}, {rr.p95_pr_auc:+.4f}] -> holdout {pos_of('pr_auc')} | dBrier 8 weeks "
                             f"{rr.full_brier:+.4f}, middle 90% [{rr.p05_brier:+.4f}, {rr.p95_brier:+.4f}] -> holdout {pos_of('brier')}")
            lines.append(f"  hindsight bound vs best baseline configuration on the holdout ({ev.oracle_pr} / {ev.oracle}): "
                         f"dPR-AUC {iv(d(STRICT_PR, 'pr_auc'))} | dBrier {iv(d(STRICT_BRIER, 'brier'))}")
            print(lines[-1], flush=True)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new", type=Path, help="the new snapshot (volunteer file)")
    mode.add_argument("--simulate-days", type=float, help="dry run: the last N days of the current snapshot as the holdout")
    mode.add_argument("--reference", action="store_true", help="spread over 3-day windows of the old test block -> reference_spread.csv")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, help="also write the report here (default for --new: results/holdout/amended.txt)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reference:
        regions, end, warn = load_robust(OLD_FILE)
        assert end == OLD_END, f"the old file ends at {end}, expected {OLD_END}"
        ref = reference_spread(regions, OLD_END)
        ref.to_csv(REFERENCE, index=False, float_format="%.5f", lineterminator="\n")
        print(ref.to_string(index=False))
        return

    ref = pd.read_csv(REFERENCE) if REFERENCE.exists() else pd.DataFrame(columns=["region", "h"])
    lines = []
    if args.simulate_days:
        regions, end, warn = load_robust(OLD_FILE)
        old_end = end - pd.Timedelta(days=args.simulate_days)
        lines.append(f"DRY RUN on already seen data (code check only): holdout = last {args.simulate_days:g} days of the current snapshot")
    else:
        snap = args.new.parent / "SNAPSHOT.txt"
        lines.append("New snapshot: " + (" | ".join(snap.read_text(encoding="utf-8").split(chr(10))).strip(" |") if snap.exists() else "SNAPSHOT.txt not found"))
        regions, end, warn = load_robust(args.new)
        old_end = OLD_END
        lines += comparison_lines(compare_snapshots(pd.read_csv(OLD_FILE), pd.read_csv(args.new), OLD_END), OLD_END)
        args.out = args.out or OUT_DIR / "amended.txt"
    lines += [f"data end {end:%Y-%m-%d %H:%M:%S} UTC", *(f"WARNING {w}" for w in warn)]
    print("\n".join(lines), flush=True)
    lines += holdout(regions, end, old_end, args.boot, ref)
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
