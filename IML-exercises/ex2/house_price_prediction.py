
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


mean_row = None

def get_mean(X: pd.DataFrame):
    global mean_row
    mean_row = X.mean()

def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if y is None:
        return preprocess_test_set(X)
    y = y.astype(float).rename('price')
    X = pd.concat([X, y], axis=1)
    X.dropna()
    X.drop(columns=['id', 'sqft_lot15', 'sqft_living15', 'long', 'lat', 'date'],
                inplace=True)
    X['sqft_living'] = pd.to_numeric(X['sqft_living'],
                                     errors='coerce')
    X['yr_built'] = pd.to_numeric(X['yr_built'], errors='coerce')
    X = X.loc[(X['sqft_living'] > 0) &
              (X['yr_built'] > 0) &
              (X['price'] > 0)]




    X = X.reset_index(drop=True)
    X['floors'] = pd.to_numeric(X['floors'], errors='coerce')
    X['bathrooms'] = pd.to_numeric(X['bathrooms'], errors='coerce')
    X['yr_renovated'] = pd.to_numeric(X['yr_renovated'], errors='coerce')
    X['condition'] = pd.to_numeric(X['condition'], errors='coerce')
    X['sqft_lot'] = pd.to_numeric(X['sqft_lot'], errors='coerce')
    X['waterfront'] = pd.to_numeric(X['waterfront'], errors='coerce')
    X['bedrooms'] = pd.to_numeric(X['bedrooms'], errors='coerce')
    X['sqft_above'] = pd.to_numeric(X['sqft_above'], errors='coerce')

    X = X.loc[(X['floors'] >= 0) &
                (X['bathrooms'] >= 0) &
                (X['yr_renovated'] >= 0) &
                (X['condition'] >= 0) &
                (X['bedrooms'] >= 0) &
                (X['sqft_lot'] >= 0) &
                (X['waterfront'] >= 0) &
                (X['sqft_above'] >= 0)
                ]
    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
    get_mean(X)
    y = X["price"]
    X.drop(columns = 'price', inplace = True)
    return X, y
    # raise NotImplementedError()

def preprocess_test_set(X: pd.DataFrame):
    X.drop(columns=['id', 'sqft_lot15', 'sqft_living15', 'long', 'lat', 'date'],
           inplace=True)
    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
    for row in X:
        X[row] = X[row].astype(float)
        for i in X[row]:
            if pd.isna(i) or i < 0:
                i = mean_row[row]
    return X

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for f in X:
        corr = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))

        # Create a scatter plot with the feature and y using Plotly
        fig = px.scatter(x=X[f], y=y, trendline="ols")
        fig.update_layout(
            title=f"{f} (corr={corr:.2f})",
            xaxis_title=f,
            yaxis_title="Response",
            hovermode="closest",
            showlegend=False,
        )
        fig.write_image(output_path + f"/pearson_cor.{f}.png")
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df.drop(columns=['price']),
                                                              df['price'], 0.75)
    # raise NotImplementedError()

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    # raise NotImplementedError()

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_X = preprocess_data(test_X)

    p_samples = []
    resultsi = np.zeros(10)
    loss_mean = np.zeros(91)
    loss_var = np.zeros(91)
    for p in range(10, 101):
        for j in range(10):
            trainX = train_X.sample(frac=p / 100.0)
            trainY = train_y.loc[trainX.index]
            resultsi[j] = LinearRegression().fit(trainX, trainY).loss(test_X, test_y)
        loss_mean[p-10] = resultsi.mean()
        loss_var[p-10] = resultsi.std()
        resultsi = np.zeros(10)

    fig = go.Figure([go.Scatter(x=list(range(10, 101)), y=loss_mean - 2 * loss_var, fill=None, mode="lines",
                                line=dict(color="lightgrey")),
                     go.Scatter(x=list(range(10, 101)), y=loss_mean + 2 * loss_var, fill='tonexty',
                                mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=list(range(10, 101)), y=loss_mean, mode="markers+lines",
                                marker=dict(color="black"))],
                    layout=go.Layout(
                        title="MSE Function",
                        xaxis=dict(title="% of Training Set"),
                        yaxis=dict(title="Test MSE"),
                        showlegend=False))
    fig.show()
    # raise NotImplementedError()