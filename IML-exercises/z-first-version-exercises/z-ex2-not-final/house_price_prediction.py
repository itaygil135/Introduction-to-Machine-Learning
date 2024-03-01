
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


# global variables
pio.templates.default = "simple_white"
changed_counter = 0
results = {}
train_columns = None


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
    # If it is the training set, remove all irrelevant rows (that are missing values):
    global results, train_columns

    process_x = pd.DataFrame.copy(X)
    process_x['date'] = pd.to_numeric(process_x['date'].str[:8], errors='coerce', downcast='integer')
    if y is None:
        for column_index in process_x.columns:
            process_x[column_index] = process_x[column_index].replace(  np.nan,
                                                                        results.get(column_index, 0))
            process_x[column_index] = process_x[column_index].apply(replace_nonsensical_values,
                                                                    args=(column_index,))
    else:
        process_y = pd.DataFrame.copy(y)
        process_x = process_x.dropna()
        process_y = process_y.loc[process_x.index]
        results = {column_index: process_x[column_index].mean() for column_index in process_x.columns}
        mask = pd.Series(True, index=process_x.index)
        for column_index in process_x.columns:
            mask &= sensical_values_indices(process_x,
                                            column_index)
        process_x = process_x[mask]
        process_y = process_y[mask]

    process_x['age'] = process_x.apply(lambda x: x.date // 10_000 - x.yr_built, axis=1)
    process_x['age_renovated'] = process_x.apply(
        lambda x: x.age if x.yr_renovated == 0 else x.date // 10_000 - x.yr_renovated, axis=1)
    # dropping features with low correlation or which we switched:
    process_x = process_x.drop(['date',
                                'long',
                                'id',
                                'lat',
                                'sqft_lot15',
                                'yr_renovated',
                                'yr_built'], axis=1)
    process_x['zipcode'] = process_x['zipcode'].astype(int)
    process_x = pd.get_dummies( process_x,
                                prefix='zipcode_',
                                columns=['zipcode'])

    if y is None:
        process_x = process_x.reindex(columns=train_columns, fill_value=0)
    else:
        train_columns = process_x.columns
        return process_x, process_y
    return process_x[train_columns]


def replace_nonsensical_values(x, column):
    global results
    global train_columns
    if np.isnan(x):
        return results.get(column, 0)
    if column in ['sqft_living', 'sqft_avobe', 'yr_built'] and x <= 0:
        return results.get(column, 0)
    if column in ['floors', 'sqft_basement', 'yr_renovated'] and x < 0:
        return results.get(column, 0)
    if column == 'waterfront' and (x < 0 or x > 1):
        return results.get(column, 0)
    if column == 'condition' and (x < 0 or x > 6):
        return results.get(column, 0)
    if column == 'view' and (x < 0 or x > 4):
        return results.get(column, 0)
    if column == 'grade' and (x < 0 or x > 14):
        return results.get(column, 0)
    if column == 'bedrooms' and (x < 0 or x > 20):
        return results.get(column, 0)
    if column == 'sqft_lot' and (x < 0 or x > 1_250_000):
        return results.get(column, 0)
    if column == 'zipcode' and x <= 0:
        return 98003  # this is just a random one
    return x


def sensical_values_indices(X: pd.DataFrame, column: str):
    if column in ['sqft_living', 'sqft_avobe', 'yr_built']:
        return X[column] > 0
    if column in ['floors', 'sqft_basement', 'yr_renovated']:
        return X[column] >= 0
    if column == 'waterfront':
        return X[column].isin(range(2))
    if column == 'view':
        return X[column].isin(range(5))
    if column == 'condition':
        return X[column].isin(range(1, 6))
    if column == 'grade':
        return X[column].isin(range(1, 16))
    if column == 'bedrooms':
        return X[column].isin(range(20))
    if column == 'sqft_lot':
        return X[column] > 0
    if column == 'long':
        return X[column] < 0
    return X[column] >= 0


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
    X.dropna(inplace=True)
    y = y.loc[X.index]

    columns_list = []
    coals = []
    for column_index in ['long','sqft_living']:
        columns_list.append(column_index)
        A = X[column_index].astype(float)
        B = y.cov(A) / (y.std() * A.std())
        coals.append(round(B, 3))
        both = pd.concat([A, y], axis=1)
        figure_item = px.scatter(   both,
                            x=column_index,
                            y='price',
                            title=f'Correlation between {column_index} and response'
                               f'\nPearson correlation: {round(B, 3)}')
        figure_item.write_image(f'{output_path}_{column_index}.png')


if __name__ == '__main__':

    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    # any prices that are Nan are irrelevant (both train and test):
    df.dropna(subset=['price'], inplace=True)
    # split to X and y:
    df_response = df['price']
    df = df.drop('price', axis=1)

    # Question 1 - split data into train and test sets
    train_x, train_y, test_x, test_y = split_train_test(df, df_response)
    # Question 2 - Preprocessing of housing prices dataset
    train_x, train_y = preprocess_data(train_x, train_y)
    test_x = preprocess_data(test_x)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(df, df_response, output_path='../')

    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    #
    lost = np.zeros((len(range(10, 101)), 10))
    for percent in range(lost.shape[0]):
        print('percent: ', percent + 10)
        for sample in range(lost.shape[1]):
            X, y, _, _ = split_train_test(train_x, train_y, ((percent + 10) / 100))
            linear_model = LinearRegression(include_intercept=True)
            linear_model.fit(X, y)
            loss = linear_model.loss(test_x, test_y)
            lost[percent][sample] = loss

    loss_mean, loss_std = lost.mean(axis=1), lost.std(axis=1)
    percent = list(range(10, 101))
    res = pd.DataFrame({'percent': list(range(10, 101)), 'mean': loss_mean, 'std': loss_std})
    figure_item = go.Figure([go.Scatter(x=percent, y=loss_mean, mode="markers+lines", showlegend=False),
                     go.Scatter(x=percent, y=loss_mean - 2 * loss_std, fill=None,
                                mode='lines',
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=percent, y=loss_mean + 2 * loss_std,
                                fill='tonexty', mode='lines',
                                line=dict(color="lightgrey"), showlegend=False),
                     ])
    figure_item.layout = go.Layout(xaxis=dict(  title='Percentage sampled'),
                                                yaxis=dict(title='MSE'),
                            title='MSE of fitted model as function of percentage of data fitted over')
    figure_item.write_image('linear_reg_mse_func.png')
