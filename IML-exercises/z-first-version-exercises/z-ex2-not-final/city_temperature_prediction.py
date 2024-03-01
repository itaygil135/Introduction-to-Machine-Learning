import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
mounth_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


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
    X = pd.read_csv(filename, header=0, parse_dates=True).dropna().drop_duplicates()
    # raise NotImplementedError()

    X = X[X["Year"] <= 2023]
    X = X[X["Month"] >= 1]
    X = X[X["Month"] <= 12]
    X = X[X["Year"] >= 1]
    X = X[X["Day"] <= 31]
    X = X[X["Day"] >= 1]
    X = X[X["Temp"] >= -60]

    X = X[X["Day"] == pd.DatetimeIndex(X['Date']).day]
    X = X[X["Month"] == pd.DatetimeIndex(X['Date']).month]
    X = X[X["Year"] == pd.DatetimeIndex(X['Date']).year]

    X['DayOfYear'] = X.apply(lambda row: row.Day + sum(mounth_days[:row.Month]), axis=1)

    return X


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    # raise NotImplementedError()
    data_ct = load_data("C:\huji\IML\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel = data_ct[data_ct["Country"] == "Israel"]

    board_color = px.colors.qualitative.Alphabet
    fig = px.scatter(israel, x="DayOfYear", y="Temp", color=israel['Year'].astype(str),
                     color_discrete_sequence=board_color)
    fig.show()

    israel_monthly = israel.groupby('Month')['Temp'].agg(['std']).reset_index()
    fig = px.bar(israel_monthly, x='Month', y='std', title='SD compared per month')
    fig.show()
    # raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    #month_of_country = data_ct.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
    month_of_country = data_ct.groupby(["Month", "Country"], as_index=False).agg(std=("Temp", "std"), mean=("Temp", "mean"))

    fig = px.line(month_of_country, x='Month', y='mean', color='Country',
                  title='Average temp  for each country per month', error_y='std')
    fig.show()
    #raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    train_of_x, train_of_y, test_of_x, test_of_y = split_train_test(israel.DayOfYear, israel.Temp)
    loss = []
    for k in range(1, 11):
        approximate_sample = PolynomialFitting(k)
        arg_x = train_of_x.to_numpy().flatten()
        arg_y = train_of_y.to_numpy().flatten()
        approximate_sample.fit(arg_x, arg_y)
        loss.append(round(approximate_sample.loss(arg_x, arg_y), 2))
    i = 0
    for val in enumerate(loss):
        print("error found for k=" + str(i + 1) + " is " + str(val))
        i = i+1
    #to change if needed
    fig = px.bar(x=range(1, 11), y=loss, title="Loss for k in range 10",
                 labels={"x": "dgree", "y": "MSE"})
    fig.show()
    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    p_loss = []
    approximate_sample = PolynomialFitting(5)
    approximate_sample.fit(israel.DayOfYear.to_numpy(), israel.Temp.to_numpy())

    Jordan = data_ct[data_ct['Country'] == 'Jordan']
    loss = approximate_sample.loss(Jordan.DayOfYear.to_numpy(), Jordan.Temp.to_numpy())
    p_loss.append(loss)

    Holland = data_ct[data_ct['Country'] == 'The Netherlands']
    loss = approximate_sample.loss(Holland.DayOfYear.to_numpy(), Holland.Temp.to_numpy())
    p_loss.append(loss)

    RSA = data_ct[data_ct['Country'] == 'South Africa']
    loss = approximate_sample.loss(RSA.DayOfYear.to_numpy(), RSA.Temp.to_numpy())
    p_loss.append(loss)

    fig = px.bar(x=['Jordan', 'The Netherlands', 'South Africa'], y=p_loss, title="Israel error over else countries",
                 labels={"x": "country", "y": "MSE val of israel"})
    #fig.update_traces(marker_color='green')
    fig.show()
    #raise NotImplementedError()
#############################################################
# import IMLearn.learners.regressors.linear_regression
# from IMLearn.learners.regressors import PolynomialFitting
# from IMLearn.utils import split_train_test
#
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.io as pio
#
# pio.templates.default = "simple_white"
#
#
# def load_data(filename: str) -> pd.DataFrame:
#     """
#     Load city daily temperature dataset and preprocess data.
#     Parameters
#     ----------
#     filename: str
#         Path to house prices dataset
#
#     Returns
#     -------
#     Design matrix and response vector (Temp)
#     """
#     X = pd.read_csv(filename, parse_dates=['Date'])
#     # remove all rows that the temp<-50 (this isn't possible, let's face it)
#     X = X[X['Temp'] > -50]
#     # create day_of_year column
#     X['DayOfYear'] = X['Date'].dt.dayofyear
#     return X
#
#
# if __name__ == '__main__':
#     np.random.seed(0)
#     # Question 1 - Load and preprocessing of city temperature dataset
#     df = load_data("C:\huji\IML\IML.HUJI\datasets\City_Temperature.csv")
#
#     # Question 2 - Exploring data for specific country
#     df_il = df[df['Country'] == 'Israel']
#
#     # Define a palette of colors for each year
#     palette = px.colors.qualitative.Alphabet
#     # Create a scatter plot
#     fig = px.scatter(df_il, x='DayOfYear', y='Temp', color='Year',
#                      color_discrete_sequence=palette)
#     fig.write_image("../israel_avg.png", engine='orca')
#
#     df_il_monthly = df_il.groupby('Month')['Temp'].agg(['std']).reset_index()
#     fig = px.bar(df_il_monthly, x='Month', y='std', title='Standard deviation compared monthly')
#     fig.write_image('../deviation_monthly.png')
#
#     # Question 3 - Exploring differences between countries
#
#     df_country_monthly = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
#     # Create a line plot with error bars
#     fig = px.line(df_country_monthly, x='Month', y='mean', error_y='std', color='Country',
#                   title='Mean and deviation for each country by month')
#     fig.write_image('../country_monthly.png')
#
#     # Question 4 - Fitting model for different values of `k`
#
#     # split the dataset
#     x_train, y_train, x_test, y_test = split_train_test(df_il.DayOfYear, df_il.Temp)
#     losses = []
#     for k in range(1, 11):
#         poly_model = PolynomialFitting(k)
#         poly_model.fit(x_train, y_train)
#         loss = round(poly_model.loss(X=x_test, y=y_test), 2)
#         losses.append(loss)
#         print(f'Loss for k={k} : {loss}')
#     loss_df = pd.DataFrame({'k': list(range(1, 11)), 'loss': losses})
#     fig = px.bar(loss_df, x='k', y='loss', title='MSE across different k chosen')
#     fig.write_image('../poly_loss.png')
#
#     CHOSEN_K = 5
#
#     # Question 5 - Evaluating fitted model on different countries
#     chosen_model = PolynomialFitting(CHOSEN_K)
#     chosen_model.fit(x_train, y_train)
#
#     df_jordan = df[df['Country'] == 'Jordan']
#     df_nether = df[df['Country'] == 'The Netherlands']
#     df_africa = df[df['Country'] == 'South Africa']
#     country_losses = [chosen_model.loss(df_jordan.DayOfYear, df_jordan.Temp),
#                       chosen_model.loss(df_nether.DayOfYear, df_nether.Temp),
#                       chosen_model.loss(df_africa.DayOfYear, df_africa.Temp)]
#     country_losses_df = pd.DataFrame({'Country': ['Jordan', 'The Netherlands', 'South Africa'],
#                                       'loss': country_losses})
#     fig = px.bar(country_losses_df, x='Country', y='loss',
#                  title='MSE across different countries using Israel model (k=5)')
#     fig.write_image('../countries_loss.png')
# import IMLearn.learners.regressors.linear_regression
# from IMLearn.learners.regressors import PolynomialFitting
# from IMLearn.utils import split_train_test
#
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.io as pio
#
# pio.renderers.default = "firefox"
# pio.templates.default = "simple_white"
#
# m_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#
#
# def load_data(filename: str) -> pd.DataFrame:
#     """
#     Load city daily temperature dataset and preprocess data.
#     Parameters
#     ----------
#     filename: str
#         Path to house prices dataset
#     Returns
#     -------
#     Design matrix and response vector (Temp)
#     """
#     #read the data
#     X = pd.read_csv(filename, header=0, parse_dates=True).dropna().drop_duplicates()
#
#     #remove invalid data
#     X = X[X["Temp"] > -70]
#     X = X[X["Day"] > 0]
#     X = X[X["Month"] > 0]
#     X = X[X["Year"] > 0]
#     X = X[X["Day"] < 32]
#     X = X[X["Month"] < 13]
#     X = X[X["Year"] < 2023]
#     #check date fit the day month year
#     X = X[X["Day"] == pd.DatetimeIndex(X['Date']).day]
#     X = X[X["Month"] == pd.DatetimeIndex(X['Date']).month]
#     X = X[X["Year"] == pd.DatetimeIndex(X['Date']).year]
#
#     #add col day of year and culc the day
#     X['DayOfYear'] = X.apply(lambda row: row.Day + sum(m_days[:row.Month]), axis=1)
#
#     return X
#
#
# if __name__ == '__main__':
#     np.random.seed(0)
#     # Question 1 - Load and preprocessing of city temperature dataset
#     data = load_data('C:\huji\IML\IML.HUJI\datasets\City_Temperature.csv')
#     # Question 2 - Exploring data for specific country
#     israel = data[data["Country"] == "Israel"]
#     fig = px.scatter(israel, x="DayOfYear", y="Temp", color=israel['Year'].astype(str),
#                      title="Temperature in relation with day of year")
#     fig.show()
#
#     monthly_std = israel.groupby('Month')['Temp'].std().rename('SD')
#     fig = px.bar(monthly_std, title="SD of each month")
#     fig.update_traces(marker_color='salmon')
#     fig.show()
#
#     # Question 3 - Exploring differences between countries
#     month = data.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
#     month.columns = ['Country', 'Month', 'T_mean', 'T_std']
#     fig = px.line(month, x='Month', y='T_mean', color=month['Country'].astype(str), title='Average temp for month per'
#                                                                                           ' country with SD',
#                   error_y='T_std')
#     fig.show()
#
#     # Question 4 - Fitting model for different values of `k`
#     israel = data[data["Country"] == "Israel"]
#     israel_t = israel.Temp
#     israel_d = israel.DayOfYear
#     train_x, train_y, test_x, test_y = split_train_test(israel_d, israel_t)
#     loss = []
#     for k in range(1, 11):
#         estimator = PolynomialFitting(k)
#         estimator.fit(train_x.to_numpy().flatten(), train_y.to_numpy().flatten())
#         loss.append(round(estimator.loss(train_x.to_numpy().flatten(), train_y.to_numpy().flatten()), 2))
#
#     for ind, val in enumerate(loss):
#         print("test error recorded for k=" + str(ind + 1) + " is " + str(val))
#
#     fig = px.bar(x=range(1, 11), y=loss, title="Loss for k in range 10",
#                  labels={"x": "dgree", "y": "MSE"})
#     fig.update_traces(marker_color='purple')
#     fig.show()
#
#     # Question 5 - Evaluating fitted model on different countries
#     p_loss = []
#     estimator = PolynomialFitting(5)
#     estimator.fit(israel_d.to_numpy(), israel_t.to_numpy())
#
#     data_i = data[data['Country'] == 'Jordan']
#     loss = estimator.loss(data_i.DayOfYear.to_numpy(), data_i.Temp.to_numpy())
#     p_loss.append(loss)
#
#     data_n = data[data['Country'] == 'The Netherlands']
#     loss = estimator.loss(data_n.DayOfYear.to_numpy(), data_n.Temp.to_numpy())
#     p_loss.append(loss)
#
#     data_sa = data[data['Country'] == 'South Africa']
#     loss = estimator.loss(data_sa.DayOfYear.to_numpy(), data_sa.Temp.to_numpy())
#     p_loss.append(loss)
#
#     fig = px.bar(x=['Jordan', 'The Netherlands', 'South Africa'], y=p_loss, title="Israel error over else countries",
#                  labels={"x": "country", "y": "MSE val of israel"})
#     fig.update_traces(marker_color='green')
#     fig.show()
#