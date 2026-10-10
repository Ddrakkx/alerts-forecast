"""Data for the interactive demo (demo/index.html): every day of the test block in Poltava oblast.

Uses the frozen pipeline READ-ONLY (src/ is imported, nothing in src/ or scripts/ is changed): the same evaluate() call as
scripts/run_experiment.py, so the curves are the real out-of-sample walk-forward predictions of the protocol logistic model and of
the no-hindsight baseline (bar B). Writes demo/data.js (a local cache, not committed) and demo/index.html: the template with the
data inlined, one self-contained file that opens without a server.

    python demo/build_demo.py              # compute the data (a few minutes) and write the page
    python demo/build_demo.py --page-only  # rewrite the page from the cached demo/data.js after editing the template
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts_forecast.data import REGION, load_regions  # noqa: E402
from alerts_forecast.experiment import configs, evaluate, feature_index, join_features  # noqa: E402
from alerts_forecast.features import build_features, neighbors_of  # noqa: E402
from alerts_forecast.target import build_frame, main_sample, state_at  # noqa: E402

KYIV = "Europe/Kyiv"
HORIZONS = [(0.25, "15 хв"), (0.5, "30 хв"), (1.0, "1 год"), (3.0, "3 год")]
AXIS_END_MIN = 27 * 60  # the timeline shows the day and 3 hours of the next one (room for the window t..t+3h)
TEST_WEEKS = 8


def minutes_from(day_start: pd.Timestamp, ts) -> np.ndarray:
    return ((pd.DatetimeIndex(ts) - day_start) / pd.Timedelta(minutes=1)).to_numpy()


def write_page(js: str) -> None:
    template = (ROOT / "demo" / "template.html").read_text(encoding="utf-8")
    marker = "/*__DEMO_DATA__*/"
    if marker not in template:
        raise SystemExit("data marker missing in demo/template.html")
    page = template.replace(marker, js.replace("</", "<\\/"))  # keep "</script>" out of the inlined JSON
    (ROOT / "demo" / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote demo/index.html ({len(page) / 1e6:.2f} MB)")


def main() -> None:
    if "--page-only" in sys.argv:
        write_page((ROOT / "demo" / "data.js").read_text(encoding="utf-8"))
        return
    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    alerts = regions[REGION]
    nbrs = list(neighbors_of(REGION))
    lanes = [REGION, *nbrs]
    test_start = data_end.floor("D") - pd.Timedelta(weeks=TEST_WEEKS)
    val_start = test_start - pd.Timedelta(weeks=TEST_WEEKS)

    hs = [h for h, _ in HORIZONS]
    feats = build_features(regions, REGION, feature_index(alerts, data_end, hs), nbrs)
    cfg = configs(nbrs)
    preds, meta_h = {}, []
    for h, label in HORIZONS:
        sample = join_features(main_sample(build_frame(alerts, data_end, h)), feats)
        ev = evaluate(sample, h, val_start, test_start, data_end, cfg)
        preds[h] = (ev.res[ev.lr_best], ev.res[ev.bar_b], build_frame(alerts, data_end, h)["y"])
        meta_h.append({"h": h, "label": label, "model": ev.lr_best, "model_cfg": ev.chosen[ev.lr_best].split("|", 1)[1],
                       "base": ev.bar_b, "base_cfg": ev.chosen[ev.bar_b].split("|", 1)[1]})
        print(f"H={h:g}: model {ev.lr_best}, baseline {ev.bar_b}", flush=True)

    # whole Kyiv days inside the test block for which every horizon has its label window observed
    first_day = (test_start.tz_convert(KYIV) + pd.Timedelta(days=1)).normalize()
    last_day = ((data_end - pd.Timedelta(hours=max(hs))).tz_convert(KYIV) - pd.Timedelta(days=1)).normalize()
    days = pd.date_range(first_day, last_day, freq="D")

    out_days, starts_per_day = {}, {}
    for day in days:
        t_local = pd.date_range(day, periods=96, freq="15min")
        t = t_local.tz_convert("UTC")
        st = state_at(alerts, t)
        f = feats.reindex(t)
        nbr_active = [[i for i, n in enumerate(nbrs) if state_at(regions[n], t[[k]])["active"].iloc[0]] for k in range(len(t))]
        starts = pd.DatetimeIndex(alerts["started_at"])
        nxt = []
        for x in t:
            later = starts[starts > x]
            nxt.append(round(float(minutes_from(day.tz_convert("UTC"), later[:1])[0]), 1) if len(later) else None)
        day_alerts = {}
        lo, hi = day.tz_convert("UTC") - pd.Timedelta(hours=12), day.tz_convert("UTC") + pd.Timedelta(minutes=AXIS_END_MIN)
        for li, name in enumerate(lanes):
            a = regions[name]
            a = a[(a["finished_at"] > lo) & (a["started_at"] < hi)]
            s = minutes_from(day.tz_convert("UTC"), a["started_at"])
            e = minutes_from(day.tz_convert("UTC"), a["finished_at"])
            day_alerts[li] = [[round(float(x), 1), round(float(y), 1), int(n)] for x, y, n in zip(s, e, a["naive"])]
        hdata = {}
        for h, _ in HORIZONS:
            pm, pb, y = preds[h]
            in_sample = t.isin(pm.index)
            hdata[str(h)] = {
                "in": [int(v) for v in in_sample],
                "y": [int(v) for v in y.reindex(t).fillna(False)],
                "pm": [round(float(pm[x]), 4) if ok else None for x, ok in zip(t, in_sample)],
                "pb": [round(float(pb[x]), 4) if ok else None for x, ok in zip(t, in_sample)],
            }
        since = st["minutes_since_last_end"].to_numpy()
        out_days[day.strftime("%Y-%m-%d")] = {
            "alerts": day_alerts,
            "active": [int(v) for v in st["active"]],
            "since": [None if np.isnan(v) else round(float(v)) for v in since],
            "s24": [int(v) for v in f["own_starts_24h"]],
            "cty": [int(v) for v in f["cty_active_n"]],
            "nbr": nbr_active,
            "next": nxt,
            "h": hdata,
        }
        own = pd.DatetimeIndex(alerts["started_at"]).tz_convert(KYIV)
        starts_per_day[day.strftime("%Y-%m-%d")] = int(((own >= day) & (own < day + pd.Timedelta(days=1))).sum())

    default_day = max(starts_per_day, key=lambda d: (starts_per_day[d], [-ord(c) for c in d]))
    meta = {
        "region": REGION, "lanes": lanes, "horizons": meta_h, "axis_end_min": AXIS_END_MIN,
        "default_day": default_day, "starts_per_day": starts_per_day,
        "test_block": f"{test_start:%Y-%m-%d} .. {data_end:%Y-%m-%d}", "snapshot": "2f115548a8bd0816c901dbf422082091b3e5c34c",
        "pipeline": "tag holdout-freeze (src/ imported read-only)",
    }
    js = "window.DEMO = " + json.dumps({"meta": meta, "days": out_days}, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (ROOT / "demo" / "data.js").write_text(js, encoding="utf-8", newline="\n")
    write_page(js)
    print(f"wrote demo/data.js: {len(out_days)} days, {len(js) / 1e6:.2f} MB, default day {default_day} "
          f"({starts_per_day[default_day]} alert starts)")


if __name__ == "__main__":
    main()
