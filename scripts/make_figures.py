"""Figures for the README, in a light and a dark version (docs/figures/*.png).

1. Share of positive moments by month (main sample, H = 3 h) for the four oblasts, with the validation and test blocks marked.
2. PR-AUC difference of the protocol logistic model to the strict reference (best baseline configuration on the test block), per oblast and horizon.
3. The same for the Brier score.
Figures 2 and 3 are read from results/<oblast>/experiment*.txt through the summary parser, so they show exactly the numbers of the tables.

    python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_regions as S  # noqa: E402
from alerts_forecast.data import load_regions  # noqa: E402
from alerts_forecast.target import MAIN_HORIZON_H, build_frame, main_sample  # noqa: E402

OUT = ROOT / "docs" / "figures"
OBLASTS = [("Poltavska oblast", "poltavska", "Poltava"), ("Kyivska oblast", "kyivska", "Kyiv oblast"),
           ("Kharkivska oblast", "kharkivska", "Kharkiv"), ("Lvivska oblast", "lvivska", "Lviv")]
HORIZONS = [(0.25, "15 min"), (0.5, "30 min"), (1.0, "1 h"), (3.0, "3 h (main)"), (6.0, "6 h")]

# validated default palette (dataviz reference instance), slots 1-4 in fixed order
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  band="#f0efec", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 band="#262624", series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(True, color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def monthly_rates() -> tuple:
    regions, data_end = load_regions(ROOT / "data" / "raw" / "volunteer_data_en.csv")
    test_start = data_end.floor("D") - pd.Timedelta(weeks=8)
    val_start = test_start - pd.Timedelta(weeks=8)
    rates = {}
    for region, _, label in OBLASTS:
        m = main_sample(build_frame(regions[region], data_end, MAIN_HORIZON_H))
        month = m.index.tz_convert("Europe/Kyiv").tz_localize(None).to_period("M")
        g = m["y"].groupby(month)
        days = pd.Series(m.index.tz_convert("Europe/Kyiv").normalize(), index=m.index).groupby(month).nunique()
        r = (g.mean() * 100)[days >= 15]  # months with fewer than 15 days of data are left out
        rates[label] = pd.Series(r.to_numpy(), index=r.index.to_timestamp() + pd.Timedelta(days=14))
    return rates, val_start.tz_localize(None), test_start.tz_localize(None), data_end.tz_localize(None)


def fig_monthly(rates, val_start, test_start, data_end, mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.grid(axis="x", visible=False)
    ax.axvspan(val_start, test_start, color=t["band"], zorder=0, linewidth=0)
    ax.axvspan(test_start, data_end, color=t["grid"], zorder=0, linewidth=0)
    for (a, b), text in (((val_start, test_start), "validation"), ((test_start, data_end), "test")):
        ax.text(a + (b - a) / 2, 2, text, color=t["ink2"], fontsize=8, rotation=90, ha="center", va="bottom")
    for (label, r), color in zip(rates.items(), t["series"]):
        ax.plot(r.index, r.values, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round", label=label)
        ax.plot(r.index[-1], r.values[-1], "o", color=color, markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
    # end labels in ink, ordered to avoid collisions
    ends = sorted(((r.values[-1], label) for label, r in rates.items()), reverse=True)
    last_x = max(r.index[-1] for r in rates.values())
    placed = []
    for value, label in ends:
        y = value
        for p in placed:
            if abs(y - p) < 6:
                y = p - 6
        placed.append(y)
        ax.text(last_x + pd.Timedelta(days=22), y, f"{label} {value:.0f}%", color=t["ink2"], fontsize=9, va="center")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlim(min(r.index[0] for r in rates.values()) - pd.Timedelta(days=20), last_x + pd.Timedelta(days=150))
    first_x = min(r.index[0] for r in rates.values())
    ticks = [d for d in pd.date_range(first_x.to_period("M").to_timestamp(), last_x, freq="MS") if d.month in (1, 7)]
    ax.set_xticks(ticks)  # no ticks beyond the data
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Share of moments without an alert that are followed by a new alert within 3 h, by month",
                 loc="left", color=t["ink"], fontsize=11.5, pad=22)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, 1.07), ncol=4, frameon=False, fontsize=9, handlelength=1.6)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig.tight_layout()
    return fig


def strict_diffs() -> dict:
    """{oblast label: {horizon: {'model': name, 'pr_auc': (d, lo, hi), 'brier': (d, lo, hi)}}} for the protocol logistic model."""
    out = {}
    for _, slug, label in OBLASTS:
        exp = S.load(slug, "experiment")
        out[label] = {h: S.strict_model_diffs(exp[h]) for h, _ in HORIZONS if h in exp}
    return out


def fig_diffs(data, metric, mode):
    t = THEMES[mode]
    color = t["series"][0]
    fig, axes = plt.subplots(1, len(OBLASTS), figsize=(11, 3.6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    ys = list(range(len(HORIZONS)))[::-1]
    for ax, (_, _, label) in zip(axes, OBLASTS):
        style(ax, t)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=t["axis"], linewidth=1, zorder=1)
        for y, (h, _) in zip(ys, HORIZONS):
            d = data[label].get(h, {}).get(metric)
            if not d:
                continue
            v, lo, hi = d
            excludes_zero = lo > 0 or hi < 0
            ax.hlines(y, lo, hi, color=color, linewidth=2, capstyle="round", zorder=2)
            if excludes_zero:  # the emphasis goes to the cells that carry a finding
                ax.plot(v, y, "o", markersize=10, zorder=3, color=color, markeredgecolor=t["surface"], markeredgewidth=2)
            else:
                ax.plot(v, y, "o", markersize=7, zorder=3, color=color, markerfacecolor=t["surface"],
                        markeredgecolor=color, markeredgewidth=1.5)
        ax.set_title(label, color=t["ink"], fontsize=10, loc="left")
        ax.set_yticks(ys)
        ax.set_yticklabels([name for _, name in HORIZONS])
        ax.tick_params(axis="x", labelsize=8.5)
    if metric == "pr_auc":
        title = "PR-AUC: protocol logistic model minus the best baseline configuration on the test block (95% interval, week blocks)"
        hint = "model better  →"
    else:
        title = "Brier score: protocol logistic model minus the best baseline configuration on the test block (95% interval, week blocks)"
        hint = "←  model better"
    fig.suptitle(title, x=0.01, ha="left", color=t["ink"], fontsize=11.5)
    fig.text(0.01, 0.015, "filled dot: interval excludes zero   ·   open dot: interval includes zero   ·   " + hint,
             color=t["ink2"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    rates, val_start, test_start, data_end = monthly_rates()
    data = strict_diffs()
    for mode in THEMES:
        suffix = "" if mode == "light" else "_dark"
        figs = {"positive_rate_by_month": fig_monthly(rates, val_start, test_start, data_end, mode),
                "diff_pr_auc": fig_diffs(data, "pr_auc", mode),
                "diff_brier": fig_diffs(data, "brier", mode)}
        for name, fig in figs.items():
            path = OUT / f"{name}{suffix}.png"
            fig.savefig(path, facecolor=fig.get_facecolor(), metadata={"Software": None})
            plt.close(fig)
            print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
