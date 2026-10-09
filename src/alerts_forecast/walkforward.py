"""Walk-forward validation: fit on the past, predict the next block, move forward."""
import pandas as pd

from .baselines import KYIV


def fold_edges(test_start: pd.Timestamp, test_end: pd.Timestamp, fold_days: int = 7):
    """Consecutive [start, end) blocks that cover [test_start, test_end)."""
    start = test_start
    while start < test_end:
        end = min(start + pd.Timedelta(days=fold_days), test_end)
        yield start, end
        start = end


def split(sample: pd.DataFrame, fold_start, fold_end, horizon_h: int, window_days=None):
    """Train/test rows for one fold.

    Train rows need t + H <= fold_start, otherwise their label window would reach into the test block
    (purge gap = H). window_days=None means all the past, otherwise only the last window_days days.
    """
    train_end = fold_start - pd.Timedelta(hours=horizon_h)
    train = sample.loc[sample.index <= train_end]
    if window_days is not None:
        train = train.loc[train.index > train_end - pd.Timedelta(days=window_days)]
    test = sample.loc[(sample.index >= fold_start) & (sample.index < fold_end)]
    return train, test


def walk_forward(sample, models: dict, horizon_h, test_start, test_end, window_days=None, fold_days=7):
    """Out-of-sample predictions for every row in [test_start, test_end).

    models: name -> fit_predict(train, test, horizon_h). Returns y, Kyiv day and one column per model.
    """
    parts = []
    for fold_start, fold_end in fold_edges(test_start, test_end, fold_days):
        train, test = split(sample, fold_start, fold_end, horizon_h, window_days)
        if train.empty or test.empty:
            continue
        out = pd.DataFrame({"y": test["y"].astype(int)}, index=test.index)
        for name, fit_predict in models.items():
            out[name] = fit_predict(train, test, horizon_h)
        parts.append(out)
    result = pd.concat(parts)
    result["day"] = result.index.tz_convert(KYIV).strftime("%Y-%m-%d")
    return result
