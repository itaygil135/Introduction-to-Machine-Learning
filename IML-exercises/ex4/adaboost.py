

import numpy as np
from IMLearn.base import BaseEstimator
from typing import Callable, NoReturn

from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner
    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator
    self.iterations_: int
        Number of boosting iterations to perform
    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations
    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator
        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator
        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        num_sampels = X[:, 0].size
        self.models_ = list()
        self.weights_ = list()
        wi_tt = 1.0
        self.D_ = (np.ones(num_sampels) / num_sampels).astype(np.float16)
        for t in range(self.iterations_):

            current_weak_learner = self.wl_().fit(X, y * self.D_)
            ppy = current_weak_learner.predict(X)
            self.models_.append(current_weak_learner)
            self.weights_.append(wi_tt)

            # if predicted everything correctly
            if np.count_nonzero(y != ppy) == 0 or (np.count_nonzero(y != ppy) == y.size):
                break

            epsilon_t = np.sum(self.D_[y != ppy])
            wi_tt = 0.5*np.log((1 - epsilon_t) / epsilon_t)   # calculate weight of learner t

            self.weights_[t] = wi_tt

            # update sample weights (for t+1 iteration)
            d_of_t1 = self.D_ * np.exp(-1*y * wi_tt * ppy).astype(np.float16)
            d_of_t1 = d_of_t1 / np.sum(d_of_t1)      # normalize sample weights
            self.D_ = d_of_t1

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if T > self.iterations_:
            T = self.iterations_

        if T > len(self.weights_):
            T = len(self.weights_)

        if T > len(self.models_):
            T = len(self.models_)

        prediction_sum = np.zeros(X[:, 0].size)
        for t in range(T):
            prediction_sum += (self.weights_[t] * self.models_[t]._predict(X))

        return np.sign(prediction_sum)


    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        ppy = self.partial_predict(X, T)
        return misclassification_error(ppy, y)
