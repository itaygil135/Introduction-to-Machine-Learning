import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iterations
    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iterations of the algorithm
    values: List[np.ndarray]
        Recorded objective values
    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback():
        def inner_callback(**kwargs):
            values.append(kwargs["val"])
            weights.append(kwargs["weights"])
        return inner_callback

    return callback(), values, weights

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    smallest_gradient1, smallest_gradient2 = 2, 2
    top_eta_L1 = 0
    top_eta_L2 = 0

    for eta in etas:
        l1 = L1(init.copy())
        l2 = L2(init.copy())
        gd_src1 = get_gd_state_recorder_callback()
        gd_src2 = get_gd_state_recorder_callback()

        gradient1 = GradientDescent(learning_rate=FixedLR(eta), callback=gd_src1[0], out_type="best")
        gradient1.fit(l1, X=None, y=None)
        gradient2 = GradientDescent(learning_rate=FixedLR(eta), callback=gd_src2[0], out_type="best")
        gradient2.fit(l2, X=None, y=None)

        fig1 = plot_descent_path(L1, np.array(gd_src1[2]), title="L1 module has eta " + str(eta))
        fig1.show()
        fig2 = plot_descent_path(L2, np.array(gd_src2[2]), title="L2 module has eta " + str(eta))
        fig2.show()

        fig3 = go.Figure([go.Scatter(x=list(range(len(gd_src1[1]))), y=gd_src1[1], mode="markers", marker_color="gold")],
                         layout=go.Layout(title="convergence rate of L1 with eta " + str(eta),
                                          xaxis_title="iterations ", yaxis_title="GD convergence value"))
        fig3.show()
        fig4 = go.Figure([go.Scatter(x=list(range(len(gd_src2[1]))), y=gd_src2[1], mode="markers", marker_color="blue")],
                         layout=go.Layout(title="convergence rate of L2 with eta " + str(eta),
                                          xaxis_title="iterations ", yaxis_title="GD convergence value"))
        fig4.show()

        if gd_src1[1][-1] < smallest_gradient1:
            smallest_gradient1 = gd_src1[1][-1]
            top_eta_L1 = eta

        if gd_src2[1][-1] < smallest_gradient2:
            smallest_gradient2 = gd_src2[1][-1]
            top_eta_L2 = eta

    print("best loss L1 module is:   " + str(smallest_gradient1) + " with eta " + str(top_eta_L1))
    print("best loss L2 module is:   " + str(smallest_gradient2) + " with eta " + str(top_eta_L2))



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)




def fit_logistic_regression():
    from IMLearn.metrics import misclassification_error
    from IMLearn.model_selection import cross_validate
    from sklearn.metrics import roc_curve, auc
    from utils import custom

    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_reg = LogisticRegression(include_intercept=True, solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    logistic_reg._fit(X_train, y_train)
    label_prob = logistic_reg.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, label_prob)
    customers = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Arbitrary Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         marker_size=5,
                         marker_color=customers[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{Custom False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{Custom True Positive Rate (TPR)}$"))).show()

    top_alpha = round(thresholds[np.argmax(tpr - fpr)], 2)
    print("The best alpha is: " + str(top_alpha))
    logistic_reg.alpha_ = top_alpha
    logistic_reg_test_error = logistic_reg._loss(X_test, y_test)
    print("Model's test error: " + str(logistic_reg_test_error))


    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    validation_error_all_1 = []
    validation_error_all_2 = []
    for lambda_val in lambdas:
        print("Starting regularization parameter: " + str(lambda_val))
        estimator1 = LogisticRegression(include_intercept=True, penalty="l1",
                                        solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000), lam=lambda_val)
        train_error, validation_error1 = cross_validate(estimator1, X_train, y_train, misclassification_error)
        validation_error_all_1.append(validation_error1)
        estimator2 = LogisticRegression(include_intercept=True, penalty="l2",
                                        solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                        lam=lambda_val)
        train_error, validation_error2 = cross_validate(estimator2, X_train, y_train, misclassification_error)
        validation_error_all_2.append(validation_error2)

    top_lambda_1 = lambdas[np.argmin(validation_error_all_1)]
    logistic_reg_L1 = LogisticRegression(include_intercept=True, penalty="l1",
                                         solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                         lam=top_lambda_1)
    print()
    print("Best lambda L1: " + str(top_lambda_1))
    logistic_reg_test_error = logistic_reg_L1.fit(X_train, y_train)._loss(X_test, y_test)
    print("L1 model's test error: " + str(logistic_reg_test_error))
    print()
    top_lambda_2 = lambdas[np.argmin(validation_error_all_2)]
    logistic_reg_L2 = LogisticRegression(include_intercept=True, penalty="l2",
                                         solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                         lam=top_lambda_2)

    print()
    print("Best lambda L2: " + str(top_lambda_2))
    logistic_reg_test_error = logistic_reg_L2.fit(X_train, y_train)._loss(X_test, y_test)
    print("L2 model's test error: " + str(logistic_reg_test_error))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
