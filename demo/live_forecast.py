"""Live forecasts at each all-clear in Poltava oblast, as pre-registered in docs/decisions.md (decision 13).

Reads only the local eMap log written by demo/emap_logger.py (no requests to any server). At start it chooses and trains the
models exactly like the holdout runner (validation = 8 weeks before the end of our snapshot, purged by H; protocol model = best
logistic family, baseline = bar B; trained once on the snapshot). Then, once a minute, it rebuilds the live history (volunteer snapshot +
eMap raion alerts merged into oblast episodes) and, for every Poltava all-clear seen after the start, writes one forecast line.

    python demo/live_forecast.py            # watch and forecast (Ctrl+C to stop)
    python demo/live_forecast.py --report   # answers from the eMap log and a summary (does not forecast)

Files in demo/live/ (committed regularly as the record): forecasts.jsonl (never edited), outcomes.csv and summary.txt (recomputed).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

from alerts_forecast.baselines import constant_rate  # noqa: E402
from alerts_forecast.data import REGION, load_regions, merge_intervals, validate_alerts  # noqa: E402
from alerts_forecast.experiment import (  # noqa: E402
    BASE_WINDOWS, POSTHOC_FAMILIES, PRESET_BASELINES, RECENT_LEVEL, RL_WINDOWS, configs, feature_index, join_features, run_configs,
)
from alerts_forecast.features import build_features, neighbors_of  # noqa: E402
from alerts_forecast.leakage import UNKNOWN_END  # noqa: E402
from alerts_forecast.metrics import brier  # noqa: E402
from alerts_forecast.target import build_frame, main_sample, state_at  # noqa: E402
from alerts_forecast.walkforward import split  # noqa: E402
from emap_logger import flatten  # noqa: E402

LOG = ROOT / "demo" / "emap_log"
OUT = ROOT / "demo" / "live"
HORIZONS = [(0.25, "15 min"), (0.5, "30 min"), (1.0, "1 h"), (3.0, "3 h")]
LOGGER_START = pd.Timestamp("2026-10-10 12:25:09", tz="UTC")
MAX_LAG = pd.Timedelta(minutes=3)  # a forecast made later than this after the all-clear is recorded but not counted
POLTAVA_UA = "Полтавська область"
UA_TO_EN = {
    "Вінницька область": "Vinnytska oblast", "Волинська область": "Volynska oblast", "Дніпропетровська область": "Dnipropetrovska oblast",
    "Донецька область": "Donetska oblast", "Житомирська область": "Zhytomyrska oblast", "Закарпатська область": "Zakarpatska oblast",
    "Запорізька область": "Zaporizka oblast", "Івано-Франківська область": "Ivano-Frankivska oblast", "Київська область": "Kyivska oblast",
    "Кіровоградська область": "Kirovohradska oblast", "Львівська область": "Lvivska oblast", "Миколаївська область": "Mykolaivska oblast",
    "Одеська область": "Odeska oblast", "Полтавська область": "Poltavska oblast", "Рівненська область": "Rivnenska oblast",
    "Сумська область": "Sumska oblast", "Тернопільська область": "Ternopilska oblast", "Харківська область": "Kharkivska oblast",
    "Херсонська область": "Khersonska oblast", "Хмельницька область": "Khmelnytska oblast", "Черкаська область": "Cherkaska oblast",
    "Чернівецька область": "Chernivetska oblast", "Чернігівська область": "Chernihivska oblast", "м. Київ": "Kyiv City",
}  # Crimea, Sevastopol and Luhansk oblast are not series in our data


def ts(x) -> pd.Timestamp:
    return pd.Timestamp(x).tz_convert("UTC") if pd.Timestamp(x).tzinfo else pd.Timestamp(x, tz="UTC")


# ---------------------------------------------------------------- the eMap log as intervals
def load_log(log_dir: Path) -> list:
    items = []
    for p in sorted(log_dir.glob("snapshot_*.json")):
        t = pd.Timestamp(datetime.strptime(p.stem.split("_", 1)[1], "%Y-%m-%dT%H%M%S%z"))
        items.append((t, 0, "snapshot", json.loads(p.read_text(encoding="utf-8"))))
    ev = log_dir / "events.jsonl"
    if ev.exists():
        for line in ev.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                items.append((ts(e["logged_at"]), 1, "event", e))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def intervals(items: list) -> tuple:
    """(closed, open): closed = [(oblast, raion, start, end, level_at_start)], end = the poll at which the end was seen;
    open = {(oblast, raion): (start, level_at_start)} for alerts still running at the last log entry."""
    open_, closed = {}, []
    for t, _, kind, obj in items:
        if kind == "snapshot":  # a (re)start of the logger: reconcile, so nothing between two processes is lost
            cur = flatten(obj)
            for key, (start, lvl, raw) in list(open_.items()):
                c = cur.get(key)
                if not c or not c["enabled"] or c["enabled_at"] != raw:
                    closed.append((*key, start, t, lvl))
                    del open_[key]
            for key, c in cur.items():
                if c["enabled"] and key not in open_ and c["enabled_at"]:
                    open_[key] = (ts(c["enabled_at"]), c["alert_level"], c["enabled_at"])
        else:
            key = (obj["oblast"], obj["raion"])
            if obj["event"] == "start" and obj.get("started_at"):
                open_[key] = (ts(obj["started_at"]), obj["alert_level"], obj["started_at"])
            elif obj["event"] == "end" and key in open_:
                start, lvl, _ = open_.pop(key)
                closed.append((*key, start, ts(obj["ended_between"][1]), lvl))
    return closed, {k: (s, lvl) for k, (s, lvl, _) in open_.items()}


def oblast_episodes(closed: list, open_: dict, oblast_ua: str) -> pd.DataFrame:
    """Union of the raion (and oblast-level) alerts of one oblast; a running alert ends at UNKNOWN_END."""
    rows = [(s, e) for o, r, s, e, _ in closed if o == oblast_ua] + [(s, UNKNOWN_END) for (o, r), (s, _) in open_.items() if o == oblast_ua]
    if not rows:
        return pd.DataFrame({"started_at": pd.Series(dtype="datetime64[ns, UTC]"), "finished_at": pd.Series(dtype="datetime64[ns, UTC]")})
    df = pd.DataFrame(rows, columns=["started_at", "finished_at"])
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True).dt.as_unit("ns")
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True).dt.as_unit("ns")
    return merge_intervals(df)


def live_history(snapshot_regions: dict, closed: list, open_: dict) -> dict:
    out = {}
    for ua, en in UA_TO_EN.items():
        base = snapshot_regions[en][["started_at", "finished_at"]]
        both = pd.concat([base, oblast_episodes(closed, open_, ua)], ignore_index=True)
        merged = merge_intervals(both)
        merged["naive"] = False
        validate_alerts(merged)
        out[en] = merged
    return out


def poltava_all_clears(closed: list, open_: dict) -> list:
    """Moments at which Poltava oblast had no enabled entry any more (the poll at which that was seen)."""
    ep = oblast_episodes(closed, open_, POLTAVA_UA)
    return [e for e in ep["finished_at"] if e < UNKNOWN_END]


# ---------------------------------------------------------------- models
def train(regions: dict, data_end: pd.Timestamp) -> dict:
    nbrs = neighbors_of(REGION)
    hs = [h for h, _ in HORIZONS]
    feats = build_features(regions, REGION, feature_index(regions[REGION], data_end, hs), nbrs)
    cfg = configs(nbrs)
    fns = {f"{fam}|{par}|{win}": (fn, BASE_WINDOWS[win]) for fam, par, win, fn in cfg}
    fns.update({f"{RECENT_LEVEL}||{w}": (constant_rate, d) for w, d in RL_WINDOWS.items()})
    models = {}
    for h, label in HORIZONS:
        sample = join_features(main_sample(build_frame(regions[REGION], data_end, h)), feats)
        val = run_configs(sample, h, data_end - pd.Timedelta(weeks=8), data_end - pd.Timedelta(hours=h), cfg)
        vb = {c: brier(val["y"], val[c]) for c in val.columns if "|" in c}
        chosen = {}
        for c, b in vb.items():
            fam = c.split("|")[0]
            if fam not in chosen or b < vb[chosen[fam]]:
                chosen[fam] = c
        fam_b = lambda f: vb[chosen[f]]  # noqa: E731
        lr = min((f for f in chosen if f.startswith("logreg")), key=fam_b)
        bar = min((f for f in (*PRESET_BASELINES, *POSTHOC_FAMILIES) if f in chosen), key=fam_b)
        models[label] = {}
        for role, fam in (("model", lr), ("baseline", bar)):
            fn, window = fns[chosen[fam]]
            tr, _ = split(sample, data_end, data_end + pd.Timedelta(days=1), h, window)
            models[label][role] = {"name": fam, "config": chosen[fam].split("|", 1)[1], "fn": fn, "train": tr, "h": h}
        print(f"  {label}: model {lr} ({chosen[lr].split('|', 1)[1]}), baseline {bar} ({chosen[bar].split('|', 1)[1]})", flush=True)
    return models


def forecast(t: pd.Timestamp, history: dict, models: dict) -> dict:
    idx = pd.DatetimeIndex([t])
    nbrs = neighbors_of(REGION)
    f = build_features(history, REGION, idx, nbrs)
    st = state_at(history[REGION], idx)
    row = f.assign(minutes_since_last_end=st["minutes_since_last_end"].to_numpy(), y=False, prev_naive=False)
    out = {}
    for label, m in models.items():
        out[label] = {role: {"name": r["name"], "config": r["config"], "p": round(float(r["fn"](r["train"], row, r["h"])[0]), 4)}
                      for role, r in m.items()}
    known = {"active_at_t": bool(st["active"].iloc[0]), "minutes_since_last_end": None if np.isnan(st["minutes_since_last_end"].iloc[0]) else float(st["minutes_since_last_end"].iloc[0]),
             "own_starts_24h": int(f["own_starts_24h"].iloc[0]), "neighbours_active": int(f["nbr_active_n"].iloc[0]),
             "oblasts_active": int(f["cty_active_n"].iloc[0])}
    return {"forecasts": out, "known_at_t": known}


def history_status(t: pd.Timestamp) -> dict:
    return {"gap": "no data 2026-10-09 05:12:41 .. 2026-10-10 12:25:09 UTC",
            "complete_24h": bool(t >= LOGGER_START + pd.Timedelta(hours=24)), "complete_7d": bool(t >= LOGGER_START + pd.Timedelta(days=7))}


# ---------------------------------------------------------------- answers from the eMap log
def last_successful_poll(polls_csv: Path) -> pd.Timestamp:
    """Time of the last poll that got the feed (200 or 304). Up to then the log is complete."""
    last = None
    for line in polls_csv.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) > 1 and parts[1] in ("200", "304"):
            last = parts[0]
    return ts(last) if last else pd.Timestamp(0, tz="UTC")


def report() -> None:
    fpath = OUT / "forecasts.jsonl"
    if not fpath.exists():
        print("no forecasts yet")
        return
    recs = [json.loads(x) for x in fpath.read_text(encoding="utf-8").splitlines() if x.strip()]
    closed, open_ = intervals(load_log(LOG))
    starts = [(s, lvl) for o, r, s, e, lvl in closed if o == POLTAVA_UA] + [(s, lvl) for (o, r), (s, lvl) in open_.items() if o == POLTAVA_UA]
    last_poll = last_successful_poll(LOG / "polls.csv")  # not the last event: quiet minutes write no event
    rows = []
    for rec in recs:
        t = ts(rec["t"])
        for label, h in [(lb, h) for h, lb in HORIZONS]:
            end = t + pd.Timedelta(hours=h)
            known = last_poll >= end + pd.Timedelta(minutes=2)
            any_ = any(t < s <= end for s, _ in starts)
            red = any(t < s <= end and lvl == "red" for s, lvl in starts)
            fc = rec["forecasts"][label]
            rows.append({"t": rec["t"], "H": label, "counted": rec["counted"], "complete_24h": rec["history"]["complete_24h"],
                         "p_model": fc["model"]["p"], "p_baseline": fc["baseline"]["p"],
                         "emap_any": int(any_) if known or any_ else None, "emap_red": int(red) if known or red else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "outcomes.csv", index=False)
    lines = [f"report at {datetime.now(timezone.utc).isoformat(timespec='seconds')}, last log poll {last_poll}", ""]
    for label, _ in [(lb, h) for h, lb in HORIZONS]:
        for truth in ("emap_any", "emap_red"):
            d = df[(df.H == label) & df.counted & df[truth].notna()]
            if d.empty:
                lines.append(f"{label:7s} {truth}: no answered forecasts yet")
                continue
            y = d[truth].astype(float)
            closer = int((abs(d.p_model - y) < abs(d.p_baseline - y)).sum())
            lines.append(f"{label:7s} {truth}: n={len(d)}, alerts={int(y.sum())}, Brier model {brier(y, d.p_model):.4f}, baseline {brier(y, d.p_baseline):.4f}, "
                         f"model closer in {closer} of {len(d)}")
    lines += ["", "Illustration only (decision 13): too few events for any significance. Primary answers (volunteer data) come later."]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------- main loop
def watch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = pd.Timestamp.now(tz="UTC")
    print(f"forecaster started {started.isoformat(timespec='seconds')}; all-clears before this are not used", flush=True)
    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    missing = set(UA_TO_EN.values()) - set(regions)
    if missing:
        raise SystemExit(f"names not in the volunteer data: {missing}")
    print("training (selection on the 8 weeks before the snapshot end, as in the holdout runner)...", flush=True)
    models = train(regions, data_end)
    state_path = OUT / "state.json"
    done = set(json.loads(state_path.read_text(encoding="utf-8"))["done"]) if state_path.exists() else set()
    print(f"watching {LOG} once a minute", flush=True)
    while True:
        try:
            closed, open_ = intervals(load_log(LOG))
            new = [e for e in poltava_all_clears(closed, open_) if e >= started and e.isoformat() not in done]
            if new:
                history = live_history(regions, closed, open_)
                for t in sorted(new):
                    made = pd.Timestamp.now(tz="UTC")
                    rec = {"t": t.isoformat(), "made_at": made.isoformat(timespec="seconds"), "lag_s": round((made - t).total_seconds()),
                           "counted": bool(made - t <= MAX_LAG), "history": history_status(t), **forecast(t, history, models)}
                    with open(OUT / "forecasts.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add(t.isoformat())
                    state_path.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
                    p = " | ".join(f"{lb}: model {v['model']['p']:.0%} vs baseline {v['baseline']['p']:.0%}" for lb, v in rec["forecasts"].items())
                    print(f"[all-clear {t:%H:%M:%S} UTC, made after {rec['lag_s']} s{'' if rec['counted'] else ', NOT counted (late)'}] {p}", flush=True)
        except Exception as e:  # keep watching; the error is visible in the terminal
            print(f"error: {type(e).__name__}: {e}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else watch()
