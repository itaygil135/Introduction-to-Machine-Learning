import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df['DayOfYear'] = df["Date"].dt.dayofyear
    df = df.loc[(df['Temp'] > 0)]
    return df
    

    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # raise NotImplementedError()
    # Question 2 - Exploring data for specific country
    df_israel = df.loc[df.Country == 'Israel']

    fig = px.scatter(df_israel, x='DayOfYear', y='Temp', color='Year',
                     color_discrete_sequence=px.colors.qualitative.Set1)
    fig.show()
    monthly_std = df_israel.groupby('Month')['Temp'].std().reset_index()

    # Create a bar plot using plotly.express
    fig = px.bar(monthly_std, x='Month', y='Temp', color='Month',
                 labels={
                     "Temp": "err"
                 },
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.show()
    # raise NotImplementedError()

    # Question 3 - Exploring differences between countries


    grouped_data = df.groupby(["Country", "Month"]).agg(mean=("Temp", "mean"), std=("Temp", "std")).reset_index()

    fig = px.line(grouped_data, x='Month', y='mean', color='Country',
                  error_y='std',
                  title='Average Monthly Temperature by Country')
    fig.show()
    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(
        df_israel['DayOfYear'], df_israel['Temp'], 0.75)

    loss_arr = np.zeros(10)
    for k in range(1,11):
        poly_obj = PolynomialFitting(k)
        poly_obj.fit(train_X, train_y)
        loss_arr[k-1] = round(poly_obj.loss(test_X, test_y), 2)
    print(loss_arr)
    fig = px.bar(loss_arr, x=[i for i in range(1, 11)], y=loss_arr,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.show()

    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    countries = df['Country'].unique()
    countries = countries[countries != 'Israel']
    poly_obj = PolynomialFitting(5)
    poly_obj.fit(df_israel['DayOfYear'], df_israel['Temp'])
    err_array = dict()
    for country in countries:
        country_df = df.loc[df.Country == country]
        err_array[country] = poly_obj.loss(country_df['DayOfYear'], country_df['Temp'])
    fig = px.bar(loss_arr, x=err_array.keys(), y=err_array.values(),
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.show()
    # raise NotImplementedError()