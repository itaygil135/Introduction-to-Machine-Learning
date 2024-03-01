

from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_error = np.inf
        thr_best = -np.inf
        feature_bestt = 0
        goodest_sign = 1
        num_features = X[0, :].size
        for feat in range(num_features):
            for siman in [-1, 1]:
                thr, thr_err = self._find_threshold(X[:, feat], y, siman)
                if thr_err < best_error:
                    feature_bestt, goodest_sign, thr_best, best_error = feat, siman, thr, thr_err

        self.threshold_, self.j_, self.sign_ = thr_best, feature_bestt, goodest_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feat vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feat

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feat vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        values = values.reshape((values.size,))
        labels = labels.reshape((labels.size,))

        # amount of true labels if the threshold is sorted_values[0] - 1  , #{i | labels[i]!=sign}
        concat = np.concatenate((values[:, np.newaxis], labels[:, np.newaxis]), axis=1)
        sorted_concat = concat[concat[:, 0].argsort()]

        sorted_values, sorted_labels = sorted_concat[:, 0], sorted_concat[:, 1]

        # Loss for classifying all as `sign` - namely, if threshold is smaller than values[0]
        leftiest_loss = np.sum(np.abs(sorted_labels)[np.sign(sorted_labels) == sign])
        if leftiest_loss <= 0:  # if all labels are equal to sign
            return -np.inf, leftiest_loss

        # labels whose sign is equal to sign will be positive, non equal will be negative
        weighted_true_labels = sorted_labels * sign

        all_losses = leftiest_loss - np.cumsum(weighted_true_labels)
        rightest_loss = all_losses[-1]

        # index for which the loss is minimal
        smallest_index = np.argmin(all_losses)

        if smallest_index >= sorted_values.size:
            return np.inf, rightest_loss

        if leftiest_loss <= all_losses[smallest_index]:
            return -np.inf, leftiest_loss

        if rightest_loss <= all_losses[smallest_index]:
            return np.inf, rightest_loss

        return sorted_values[smallest_index], all_losses[smallest_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

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
        return misclassification_error(self._predict(X), y)

