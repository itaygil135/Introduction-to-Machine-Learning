
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree
    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate
    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions
    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate
    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    num_rows = X.shape[0]  # Get the number of rows in the array
    random_indices = np.random.choice(num_rows, size=n_samples, replace=False)  # Generate 50 random unique indices

    train_samples = X[random_indices]
    train_responses = y[random_indices]
    test_samples = X[~random_indices]
    test_responses = y[~random_indices]

    # n_evaluations = 2
    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    r_lamda = np.linspace(0.0001, 0.1, num=n_evaluations)
    l_lamda = np.linspace(0.01, 2, num=n_evaluations)
    r_train_err = list()
    r_val_err = list()
    l_train_err = list()
    l_val_err = list()
    for lam in range(n_evaluations):
        ridge_train_error, ridge_val_error = cross_validate(RidgeRegression(r_lamda[lam]),
                                                            train_samples, train_responses, mean_square_error)
        r_train_err.append(ridge_train_error)
        r_val_err.append(ridge_val_error)

        lasso_train_error, lasso_val_error = cross_validate(Lasso(l_lamda[lam]),
                                                            train_samples, train_responses, mean_square_error)
        l_train_err.append(lasso_train_error)
        l_val_err.append(lasso_val_error)





    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["ridge", "Lasso"],
                        horizontal_spacing=0.01, vertical_spacing=.1)

    fig.add_traces([go.Scatter(x=r_lamda, y=r_train_err, fill=None, name="Ridge train loss", mode="lines",
                               line=dict(color="blue")),
                    go.Scatter(x=r_lamda, y=r_val_err, fill=None, name="Ridge test loss", mode="lines",
                               line=dict(color="yellow"), showlegend=True)],
                   rows=1, cols=1).update_xaxes(title="Lamda")
    fig.add_traces([go.Scatter(x=l_lamda, y=l_train_err, fill=None,name="Lasso train loss", mode="lines",
                               line=dict(color="red")),
                    go.Scatter(x=l_lamda, y=l_val_err, fill=None, name="Lasso train loss", mode="lines",
                               line=dict(color="green"), showlegend=True)],
                   rows=1, cols=2).update_xaxes(title="Lamda")
    fig.update_layout(title="Lost as a function of Lambda")

    fig.show()


    # # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    index_ridge = np.argmin(r_val_err)
    ridge_best_l = r_lamda[index_ridge]
    print("Best lambda for ridge is: ", ridge_best_l)

    index_lasso = np.argmin(l_val_err)
    lasso_l = l_lamda[index_lasso]
    print("Best lambda for lasso is: ", lasso_l)

    ridge_result = RidgeRegression(lam=ridge_best_l).fit(train_samples, train_responses)
    ridge_result_error = ridge_result.loss(test_samples, test_responses)
    print("Best ridge loss is: ", ridge_result_error)

    lasso_reasult = Lasso(alpha=lasso_l)
    lasso_reasult.fit(train_samples, train_responses)
    lasso_reasult_error = mean_square_error(lasso_reasult.predict(test_samples), test_responses)
    print("Best lasso loss is: ", lasso_reasult_error)

    linear_r = LinearRegression().fit(train_samples, train_responses)
    linear_r_loss = linear_r.loss(test_samples, test_responses)
    print("Linear Regression loss is: ", linear_r_loss)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
