

import numpy as np
from typing import Tuple
# from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.metalearners.adaboost import AdaBoost

from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    Our_learner = AdaBoost(DecisionStump, n_learners)
    Our_learner.fit(train_X, train_y)

    train_loss = []
    real_loss =[]
    for i in range(1, n_learners):
        train_loss.append(Our_learner.partial_loss(train_X, train_y, i))
        real_loss.append(Our_learner.partial_loss(test_X, test_y, i))

    fig = go.Figure(
        data=[go.Scatter(x=[i for i in range(1, n_learners)], y=train_loss,
                         name="Train", mode="lines"),
              go.Scatter(x=[i for i in range(1, n_learners)], y=real_loss,
                         name="Test", mode="lines")],
        layout=go.Layout(
            title="Train and test error",
            xaxis_title="num of models",
            yaxis_title="Error"))
    fig.show()

    # raise NotImplementedError()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    limit = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise == 0:
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=[rf"$\textbf{{{m}}}$" for m in T],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, t in enumerate(T):
            fig.add_traces([decision_surface(lambda X:Our_learner.partial_predict(X,t),
                                             limit[0], limit[1], showscale=False),
                            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                       showlegend=False,
                                       marker=dict(color=test_y, symbol=["circle" if i == 1 else "x" for i in test_y],
                                                   colorscale=[custom[0],
                                                               custom[-1]],
                                                   line=dict(color="black",
                                                             width=1)))],
                           rows=(i // 2) + 1, cols=(i % 2) + 1)

        fig.update_layout(
            title="Decision surface for different iterations ",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()
    # raise NotImplementedError()

    # Question 3: Decision surface of best performing ensemble
    if noise == 0:
        min_loss = float('inf')
        ens_size = 0
        for i in range(1,n_learners):
            curr_loss = Our_learner.partial_loss(test_X, test_y, i)
            if curr_loss < min_loss:
                min_loss = curr_loss
                ens_size = i

        fig = go.Figure([
            decision_surface(lambda X: Our_learner.partial_predict(X, ens_size), limit[0],
                             limit[1]),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       marker=dict(color=test_y,
                                   symbol=["circle" if i == 1 else "x" for i in test_y]))],
            layout=go.Layout(
                             title=f"Best model decision surface "))

        fig.show()
        print("Ensemble Size: ", ens_size)
        print("Accuracy : ", 1 - round(real_loss[ens_size - 1], 4))
    # raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    D = (Our_learner.D_ / Our_learner.D_.max())*15
    fig = go.Figure([
        decision_surface(Our_learner.predict, limit[0], limit[1]),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", marker=dict(size=D, color=train_y,
                               symbol=["circle" if i == 1 else "x" for i in test_y]))],
        layout=go.Layout(title=f"Last Iteration Decision surface "))
    fig.show()
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
    # raise NotImplementedError()
