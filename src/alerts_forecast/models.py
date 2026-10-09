"""Probabilistic models with the same fit_predict(train, test, horizon_h) interface as the baselines."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(columns: list, C: float):
    """Standardised logistic regression on the given columns. The scaler is fit on the training rows only."""

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, horizon_h: int) -> np.ndarray:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        pipe.fit(train[columns], train["y"].astype(int))
        return pipe.predict_proba(test[columns])[:, 1]

    return fit_predict


def make_platt(base_fit_predict, cal_days: int = 28):
    """Recalibrate a model with Platt scaling fitted on the most recent cal_days of the training window.

    The model itself is fit on the older part only (with a gap of H hours so labels do not overlap),
    the newest part of the past teaches the sigmoid. Test labels are never used.
    """

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, horizon_h: int) -> np.ndarray:
        cal_start = train.index.max() - pd.Timedelta(days=cal_days)
        fit_part = train.loc[train.index <= cal_start - pd.Timedelta(hours=horizon_h)]
        cal_part = train.loc[train.index > cal_start]
        both = base_fit_predict(fit_part, pd.concat([cal_part, test]), horizon_h)
        p_cal, p_test = both[: len(cal_part)], both[len(cal_part):]
        eps = 1e-4
        logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)  # noqa: E731
        sigmoid = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_cal), cal_part["y"].astype(int))
        return sigmoid.predict_proba(logit(p_test))[:, 1]

    return fit_predict


def make_hgb(columns: list, max_depth: int, n_iter: int):
    """Small gradient-boosted trees. Strong limits on purpose: signal is weak and neighbouring rows are near-duplicates.

    No early stopping (sklearn would split off a random, not time-ordered, validation part of the training rows).
    """

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, horizon_h: int) -> np.ndarray:
        model = HistGradientBoostingClassifier(
            max_depth=max_depth, max_iter=n_iter, learning_rate=0.05, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=0,
        )
        model.fit(train[columns], train["y"].astype(int))
        return model.predict_proba(test[columns])[:, 1]

    return fit_predict
