"""Probabilistic metrics and a day-block bootstrap for their uncertainty."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def reliability(y, p, bins: int = 5) -> pd.DataFrame:
    """Calibration table: rows are quantile bins of the predicted probability."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    return out.round(3)


def calibration_slope_intercept(y, p, eps: float = 1e-4) -> tuple[float, float]:
    """Fit y ~ logit(p). Slope 1 and intercept 0 mean perfectly calibrated; slope < 1 means overconfident."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y, int))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def with_blocks(result: pd.DataFrame, unit: str = "week") -> pd.DataFrame:
    """Set the bootstrap block column ('day'): 'day' keeps Kyiv days, 'week' = ISO weeks of the Kyiv calendar
    (the alert rate persists for weeks, so day blocks understate the uncertainty), '6h' = 6-hour blocks (short holdout)."""
    local = result.index.tz_convert("Europe/Kyiv")
    if unit == "day":
        return result.assign(day=local.strftime("%Y-%m-%d"))
    if unit == "week":
        return result.assign(day=local.strftime("%G-%V"))
    if unit == "6h":
        return result.assign(day=[f"{d}-{h // 6}" for d, h in zip(local.strftime("%Y-%m-%d"), local.hour)])
    raise ValueError(f"unknown block unit {unit!r}")


def day_block_bootstrap(result: pd.DataFrame, models: list, reference, n_boot: int = 1000, seed: int = 0):
    """Resample whole days (rows of one day stay together) to respect dependence between neighbours.

    result: y, day and one probability column per model.
    reference: one model name or a list of names. Returns (table, diffs): point estimate and 95% interval
    per model and metric, and the paired difference to every reference model on the same resampled days.
    """
    references = [reference] if isinstance(reference, str) else list(reference)
    y = result["y"].to_numpy()
    probs = {m: result[m].to_numpy() for m in models}
    _, inverse = np.unique(result["day"].to_numpy(), return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)

    metrics = {"pr_auc": pr_auc, "brier": brier}
    draws = {(m, k): [] for m in models for k in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].min() == y[idx].max():
            continue
        for m in models:
            for k, fn in metrics.items():
                draws[(m, k)].append(fn(y[idx], probs[m][idx]))

    rows, diffs = [], []
    for m in models:
        for k, fn in metrics.items():
            d = np.array(draws[(m, k)])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"model": m, "metric": k, "value": fn(y, probs[m]), "lo": lo, "hi": hi})
            for ref in references:
                if m == ref:
                    continue
                dd = d - np.array(draws[(ref, k)])
                better = (dd > 0) if k == "pr_auc" else (dd < 0)
                dlo, dhi = np.percentile(dd, [2.5, 97.5])
                diffs.append(
                    {
                        "model": m, "metric": k, "vs": ref,
                        "diff": fn(y, probs[m]) - fn(y, probs[ref]),
                        "lo": dlo, "hi": dhi, "share_better": float(better.mean()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(diffs)
