"""Collect the headline numbers of every region from the saved result files into results/summary.md.

Nothing is recomputed: the tables are parsed from results/<region>/experiment*.txt and boosting*.txt, so the
summary can be regenerated and compared with the source files.

    python scripts/summarize_regions.py
"""
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
NL = chr(10)
REGIONS = [("poltavska", "Poltavska"), ("kyivska", "Kyivska"), ("kharkivska", "Kharkivska"), ("lvivska", "Lvivska")]
MODEL = "logreg[own+nbr+cty]"  # used only in the neighbour table (same model with vs without the features)
STRICT_BRIER, STRICT_PR = "STRICT_BRIER", "STRICT_PR"
BOOST = "hgb[own+nbr+cty]"
MAIN_H = (1, 3, 6)
SHORT_H = (0.25, 0.5)  # additional (decision 8)

ROW = re.compile(r"^\s*(?P<fam>\S+)(?: \*post-hoc)?\s+(?P<chosen>\S*\|\S+)\s+(?P<val>[\d.]+)\s+(?P<brier>[\d.]+) "
                 r"\[(?P<blo>[\d.]+), (?P<bhi>[\d.]+)\]\s+(?P<pr>[\d.]+) \[(?P<plo>[\d.]+), (?P<phi>[\d.]+)\]")
DIFF = re.compile(r"^\s*(?P<model>\S+)\s+(?P<metric>pr_auc|brier)\s+(?P<d>-?[\d.]+) \[(?P<lo>-?[\d.]+), (?P<hi>-?[\d.]+)\]")


def hlabel(h: float) -> str:
    return f"{round(h * 60)} min" if h < 1 else f"{h:g} h"


def sections(text: str) -> dict:
    """{horizon: lines} by the '===== H = n h' headers."""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"=+ H = ([\d.]+) h", line)
        if m:
            cur = float(m.group(1))
            out[cur] = [line]  # keep the header: boosting prints its reference bar there
        elif cur is not None:
            out[cur].append(line)
    return out


def load(slug: str, stem: str) -> dict:
    """Sections of results/<slug>/<stem>.txt and, if present, <stem>_short.txt."""
    out = {}
    for name in (f"{stem}.txt", f"{stem}_short.txt"):
        path = RESULTS / slug / name
        if path.exists():
            out.update(sections(path.read_text(encoding="utf-8")))
    return out


def diffs(lines, ref: str, model: str) -> dict:
    """Paired differences of `model` to `ref` from the block 'Paired difference to <ref>'."""
    res, inside = {}, False
    for line in lines:
        if line.startswith("Paired difference to") or line.startswith("POST HOC"):
            inside = line.startswith("Paired") and line.split()[3].rstrip(";") == ref
            continue
        if inside:
            m = DIFF.match(line)
            if m and m["model"] == model:
                res[m["metric"]] = (float(m["d"]), float(m["lo"]), float(m["hi"]))
            elif not line.strip() and res:
                inside = False
    return res


def posthoc(lines, vs: str) -> dict:
    """From the boosting POST HOC block: difference of hgb[own+nbr+cty] to the family `vs`."""
    res, inside = {}, False
    for line in lines:
        if line.startswith("POST HOC"):
            inside = True
            continue
        if inside:
            m = DIFF.match(line)
            if m and m["model"] == vs:
                res[m["metric"]] = (float(m["d"]), float(m["lo"]), float(m["hi"]))
    return res


def protocol_model(lines) -> str:
    """The logistic family the protocol chose on validation (lr_best) for this oblast and horizon."""
    return re.search(r"protocol model \(best logistic family on validation\) = (\S+)", NL.join(lines))[1]


def strict_labels(lines) -> tuple:
    """(best baseline configuration on the test block by Brier, by PR-AUC)."""
    text = NL.join(lines)
    return (re.search(r"strict reference by Brier \(best baseline configuration on the test block\) = (\S+)", text)[1],
            re.search(r"strict reference by PR-AUC \(best baseline configuration on the test block\) = (\S+)", text)[1])


def strict_model_diffs(lines) -> dict:
    """{'model': name, 'pr_auc': (d, lo, hi), 'brier': (d, lo, hi)}: the protocol model against the strict reference of each metric."""
    model = protocol_model(lines)
    return {"model": model, "pr_auc": diffs(lines, STRICT_PR, model).get("pr_auc"),
            "brier": diffs(lines, STRICT_BRIER, model).get("brier")}


def fmt_diff(d, digits) -> str:
    if not d:
        return "n/a"
    v, lo, hi = d
    mark = " †" if (lo > 0 or hi < 0) else ""
    return f"{v:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{mark}"


def table(title: str, mode: str, horizons) -> list:
    """mode 'bar': reference chosen on validation (bar B); mode 'strict': best baseline configuration on the test block, per metric."""
    out = [f"## {title}", "",
           "| Region | H | positive in test / validation | protocol logistic | reference (Brier / PR-AUC) | its Brier | its PR-AUC "
           "| logistic: dPR-AUC | logistic: dBrier | protocol boosting | boosting: dPR-AUC | boosting: dBrier |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for slug, name in REGIONS:
        exp, bst = load(slug, "experiment"), load(slug, "boosting")
        for h in horizons:
            if h not in exp:
                continue
            lines = exp[h]
            text = NL.join(lines)
            pos = re.search(r"positive rate ([\d.]+)%; validation rows \d+, positive rate ([\d.]+)%", text)
            model = protocol_model(lines)
            if mode == "bar":  # one reference for both metrics
                ref_b = ref_p = re.search(r"bar B \(incl\. post-hoc baselines\) = (\S+)", text)[1]
                label = ref_b
            else:  # the strict reference is the best baseline configuration on the test block, per metric
                ref_b, ref_p = STRICT_BRIER, STRICT_PR
                lb, lp = strict_labels(lines)
                label = lb if lb == lp else f"{lb} / {lp}"
            rows = {m["fam"]: m for m in map(ROW.match, lines) if m}
            lg_p, lg_b = diffs(lines, ref_p, model), diffs(lines, ref_b, model)
            hg_p = hg_b = {}
            boost = "n/a"
            if h in bst:
                header = bst[h][0]
                boost = re.search(r"protocol boosting = (\S+),", header)[1]
                hb = hp = re.search(r"bar B = (\S+),", header)[1] if mode == "bar" else None
                if mode != "bar":
                    hb, hp = STRICT_BRIER, STRICT_PR
                hg_p, hg_b = diffs(bst[h], hp, boost), diffs(bst[h], hb, boost)
            out.append(f"| {name} | {hlabel(h)} | {pos[1]}% / {pos[2]}% | {model} | {label} | {float(rows[ref_b]['brier']):.4f} | {float(rows[ref_p]['pr']):.3f} | "
                       f"{fmt_diff(lg_p.get('pr_auc'), 3)} | {fmt_diff(lg_b.get('brier'), 4)} | {boost} | "
                       f"{fmt_diff(hg_p.get('pr_auc'), 3) if h in bst else 'n/a'} | {fmt_diff(hg_b.get('brier'), 4) if h in bst else 'n/a'} |")
    return out + [""]


def neighbours_table(horizons) -> list:
    """Do neighbours and the country help? Same model family with and without them, paired."""
    out = ["## Do neighbours and country-wide activity help? (same model with vs without them)", "",
           "| Region | H | logistic +nbr vs own: dPR-AUC | dBrier | logistic +nbr+cty vs own: dPR-AUC | dBrier | boosting +nbr+cty vs own: dPR-AUC | dBrier |",
           "|---|---|---|---|---|---|---|---|"]
    for slug, name in REGIONS:
        exp, bst = load(slug, "experiment"), load(slug, "boosting")
        for h in horizons:
            if h not in exp:
                continue
            nbr = diffs(exp[h], "logreg[own]", "logreg[own+nbr]")
            cty = diffs(exp[h], "logreg[own]", MODEL)
            hgb = posthoc(bst[h], "hgb[own]") if h in bst else {}
            out.append(f"| {name} | {hlabel(h)} | {fmt_diff(nbr.get('pr_auc'), 3)} | {fmt_diff(nbr.get('brier'), 4)} | "
                       f"{fmt_diff(cty.get('pr_auc'), 3)} | {fmt_diff(cty.get('brier'), 4)} | "
                       f"{fmt_diff(hgb.get('pr_auc'), 3)} | {fmt_diff(hgb.get('brier'), 4)} |")
    return out + [""]


def main() -> None:
    out = ["# Summary across oblasts (generated by scripts/summarize_regions.py, do not edit by hand)", "",
           "Test = last 8 weeks. Differences are paired, bootstrap over ISO weeks; † marks an interval that excludes zero "
           "(judged on the rounded numbers). PR-AUC: positive = better; Brier: negative = better. Models: the logistic and the boosting "
           "family that the protocol chose on validation (named in each row). Main horizon: 3 h. 15 and 30 min are additional (decision 8).", ""]
    out += table("Main horizons, against the baseline chosen on the validation block (no hindsight)", "bar", MAIN_H)
    out += table("Main horizons, against the best baseline CONFIGURATION on the test block, per metric "
                 "(strict: chosen with hindsight among about 40 configurations, favours the baselines)", "strict", MAIN_H)
    out += table("Additional short horizons, against the baseline chosen on the validation block (no hindsight)", "bar", SHORT_H)
    out += table("Additional short horizons, against the best baseline configuration on the test block, per metric (strict)", "strict", SHORT_H)
    out += neighbours_table((*SHORT_H, 1, 3))
    (RESULTS / "summary.md").write_text(NL.join(out) + NL, encoding="utf-8", newline=NL)  # LF on every system
    print(NL.join(out))


if __name__ == "__main__":
    main()
