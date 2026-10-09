"""Probabilistic models with the same fit_predict(train, test, horizon_h) interface as the baselines."""
import numpy as np
import pandas as pd
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
