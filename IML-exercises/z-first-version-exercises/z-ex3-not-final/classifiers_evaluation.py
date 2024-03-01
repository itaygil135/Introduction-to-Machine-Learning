import numpy as np
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
    Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data_set = np.load(filename)
    return data_set[:, :2], data_set[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets
    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for num, f in [("Linearly Separable", "C:\huji\IML\IML.HUJI\datasets\linearly_separable.npy"),
                 ("Linearly Inseparable", "C:\huji\IML\IML.HUJI\datasets\linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(perceptron: Perceptron, x: np.ndarray, y_: int):
            losses.append(perceptron._loss(X, y))

        perceptron = Perceptron(callback=callback)
        perceptron._fit(X, y)

        # Plot figure of loss as function of fitting iteration
        figure = px.line(x=range(len(losses)), y=losses, title=f"perceptron algorithm's training loss when data  is {num}")
        figure.update_traces(line_color='darksalmon')
        figure.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    lo_1, lo_2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(lo_1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (lo_1 * np.cos(theta) * np.cos(t)) - (lo_2 * np.sin(theta) * np.sin(t))
    ys = (lo_1 * np.sin(theta) * np.cos(t)) + (lo_2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")
def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"C:\huji\IML\IML.HUJI\datasets\{f}")

        # Fit models and predict over training set
        naive, l_d_a = GaussianNaiveBayes().fit(X, y), LDA().fit(X, y)
        naive_preds, l_d_a_preds = naive.predict(X), l_d_a.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        figure = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                            rf"$\text{{Gaussian Naive Bayes (accuracy={round(100 * accuracy(y, naive_preds), 2)}%)}}$",
                            rf"$\text{{Linear discriminant analysis (accuracy={round(100 * accuracy(y, l_d_a_preds), )}%)}}$"))

        # Add traces for data-points setting symbols and colors
        figure.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=naive_preds, symbol=class_symbols[y], colorscale=class_colors(3))),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=l_d_a_preds, symbol=class_symbols[y], colorscale=class_colors(3)))],
                       rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        figure.add_traces([go.Scatter(x=naive.mu_[:, 0], y=naive.mu_[:, 1], mode="markers", marker=dict(symbol="x", color="black", size=15)),
                        go.Scatter(x=l_d_a.mu_[:, 0], y=l_d_a.mu_[:, 1], mode="markers", marker=dict(symbol="x", color="black", size=15))],
                       rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for index in range(3):
            figure.add_traces([get_ellipse(naive.mu_[index], np.diag(naive.vars_[index])), get_ellipse(l_d_a.mu_[index], l_d_a.cov_)],
                           rows=[1, 1], cols=[1, 2])

        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        figure.update_layout(title_text=rf"$\text{{Comparing Gaussian Classifiers - {f[:-4]} dataset}}$",
                          width=800, height=400, showlegend=False)
        # figure.write_image(f"l_d_a.vs.naive.bayes.{f[:-4]}.png")
        figure.show()



if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

