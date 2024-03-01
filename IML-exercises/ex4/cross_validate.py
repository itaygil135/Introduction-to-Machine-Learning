
from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator
    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data
    X: ndarray of shape (n_samples, n_features)
       Input data to fit
    y: ndarray of shape (n_samples, )
       Responses of input data to fit to
    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.
    cv: int
        Specify the number of folds.
    Returns
    -------
    train_score: float
        Average train score over folds
    validation_score: float
        Average validation score over folds
    """
    # split all indexes to 'cv' groups - k_fold[0] contains the indexes of the samples in the first group
    k_folds = np.array_split(np.arange(X[:, 0].size), cv)   # List[ndarray]

    train_err_schoom = 0
    value_err_schoom = 0
    for fold_ids in k_folds:    # fold_ids is a ndarray containing the indexes of the current group
        mk = np.ones(X[:, 0].size, dtype=bool)
        mk[fold_ids] = False
        train_X, train_y = X[mk, :], y[mk]
        est = estimator.fit(train_X, train_y)
        train_err_schoom = train_err_schoom + scoring(est.predict(train_X), train_y)
        value_err_schoom = value_err_schoom + scoring(est.predict(X[~mk, :]), y[~mk])

    train_averge = train_err_schoom / cv
    val_averge = value_err_schoom / cv
    return train_averge, val_averge

