from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        labels_dict = {}
        for yi in y:
            if yi not in labels_dict:
                labels_dict[yi] = 1
            else:
                labels_dict[yi] += 1
        lst = list(labels_dict.keys())
        lst.sort()
        sorted_dict = {i: labels_dict[i] for i in lst}
        self.classes_ = np.array([])
        self.pi_ = np.array([])
        for yi,counter in sorted_dict.items():
            self.classes_ = np.append(self.classes_,yi)
            self.pi_ = np.append(self.pi_, counter/len(y))

        self.mu_ = np.array([np.mean(X[y == i], axis=0) for i in self.classes_])
        c = X - self.mu_[y.astype(int)]
        dim_features = c.shape[1]
        self.cov_ = np.zeros((dim_features, dim_features))

        for i in range(dim_features):
            for j in range(dim_features):
                self.cov_[i, j] = np.dot(c[:, i], c[:, j])
        self.cov_ /= c.shape[0]
        self.cov_inv_ = inv(self.cov_)
        # raise NotImplementedError()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = np.argmax(self.likelihood(X), axis=1)
        return y_pred
        # raise NotImplementedError()

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        likelihoods = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            mu = self.mu_[i]
            cov = self.cov_
            prior = self.pi_[i]

            diff = X - mu
            exponent = -0.5 * np.sum(np.dot(diff, np.linalg.inv(cov)) * diff,
                                     axis=1)
            likelihoods[:, i] = np.exp(exponent) * prior
        return likelihoods
        # raise NotImplementedError()

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
        # raise NotImplementedError()
