import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # See https://github.com/ContinuumIO/anaconda-issues/issues/905

import sys
import time
from datetime import timedelta
import pickle
import random as rand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.signal

import sqlalchemy
import pymysql
import urllib
import pendulum

import logging
logging.getLogger('parso.python.diff').disabled = True  # Stops IPython Parso logger from logging during tab-completion

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

pd.plotting.register_matplotlib_converters()  # Allows using the .plot() on Series with time data.

"""
Several helper functions and objects for use inside IPython or a Jupyter Notebook. 
At the beginning of a session, do:
    %load helpers.py
to load all of these functions and objects, as well as import all modules you will need and configure any settings

"An approximate answer to the right problem is worth a good deal more than an exact answer to an approximate problem." 
--John Tukey

"He uses statistics like a drunk person uses a lamp post, more for support than illumination." 
-- Andrew Lang

"All models are wrong, but some are useful."
-- George Box
"""


def get_msql_con(server, database, windows_authentication=True):
    """
    Returns sqlalchemy database connection on the specified server and database.

    Must use fast_executemany=True to enable better write speed, and ODBC Driver 17 is one of the few
    compatible drivers for sqlalchemy.

    :param server: String indicating which server to connect to
    :param database: String indicating which database to connect to
    :param windows_authentication: Boolean indicating whether or not to use Windows authentication
    :return: sqlalchemy engine object

    """
    if windows_authentication:
        trusted_connection = r';Trusted_Connection=yes'
    else:
        trusted_connection = ';'

    params = urllib.parse.quote_plus(
        r'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + r';DATABASE=' + database + trusted_connection)
    con = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(params), fast_executemany=True)

    return con


def get_mysql_con(host, database, user, password):
    """
    Creates MySQL database connection.

    :param host: String representing the hostname to connect to
    :param user: String representing the username to login with
    :param password: String representing the password for the given username
    :param database: String representing which database to connect to

    :return: pymysql Connection object
    """

    return pymysql.connect(host=host, user=user, password=password, db=database,
                           cursorClass=pymysql.cursors.DictCursor)


def reld(a, b):
    """
    Gets the relative difference between two values.

    :param a: The first number
    :param b: The second number
    :return: Float representing the relative difference of two numbers
    """
    return np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2)


def pctc(a, b):
    """
    Gets the percent change between two values.

    :param a: The first number
    :param b: The second number
    :return: Float representing the percent change in two numbers
    """

    return (b - a) / np.abs(a)


def get_n_percent_rows(s, n=.95):
    """
    Calculates all rows that account for the top n percent of that Series (sorted descending).
    Useful, for instance, if you wanted to find the fewest number of rows that account for 95% of a Series.

    :param s: Series to be sorted
    :param n: Float that represents the top n percent of rows to be found.
    :return:
    """

    if n >= 1:
        raise ValueError("The input percent must be less than 100.")

    s = s.sort_values(ascending=False)

    running_total = 0
    for i in range(len(s)):
        running_total += s.iloc[i]

        if running_total >= s.sum() * n:
            return s[:i]


def plot_last_year(df, ax, lags=(1, )):
    """Plots last year's y-values next to this year's y-values. Accepts a tuple of yearly lags to plot, i.e.
    a lag of one indicates plot last year's value, a lag of one and two would plot last year and the year before."""

    colors = [('red', 'violet'),
              ('darkgreen', 'limegreen'),
              ('navy', 'cornflowerblue')]

    fixed_df = df.copy()
    if df.index.name != 'ds':
        if df.index.name is None:
            drop = False
        else:
            drop = True

        fixed_df = df.reset_index(drop=drop).set_index('ds')

    for i, lag in enumerate(lags):
        mec, mfc = colors[i]
        ax.plot(fixed_df['y'].shift(364 * lag),
                ls='', marker='D', mec=mec, mfc=mfc, markersize=4 - .5 * i)


def mape(y, yhat):
    """
    Calculates the Mean Absolute Percent Error between two Series.

    :param y: Series containing the actual y values
    :param yhat: Series containing the forecasts

    :return: Float, the Mean Absolute Percent Error
    """

    yhat = yhat.loc[y.index]

    return 100 * np.mean(np.abs((yhat - y) / y))


def rmse(y, yhat):
    """
    Calculates the Root Mean Squared Error between two Series.
    :param y: Series containing the actuals
    :param yhat: Series containing the forecasts

    :return: Float, the Root Mean Squared Error
    """

    yhat = yhat.loc[y.index]

    return np.sqrt((np.square((yhat - y)).mean()))


def calc_yoy_growth(x, date, lags=3, inclusive=False):
    """
    Looks at the year-over-year growth for the past n weekly lags. If inclusive, then the given date is included as a
    lag.
    For example, look at the last 3 Fridays before a given Friday and calculate their growth over the corresponding
    Fridays in the year before. If inclusive, the growth would be the last 2 Fridays and the Friday given.

    :param x: DataFrame or Series containing dates and corresponding values.
    :param date: String or Datetime-like, TODO a list-like containing dates to use for reference points.
    :param lags: Int representing number of weekly shifts to use for growth
    :param inclusive: Bool, whether or not to use the given date's value in the calculation

    :return: Float describing the year-over-year growth.
    """

    date = pd.to_datetime(date)

    if x.index.name != 'ds':
        x.index = x.index.get_level_values('ds')

    if inclusive:
        lags = range(lags)
    else:
        lags = range(1, lags+1)

    # Pandas is weird and will return a slice using .loc with a string, but not with Timestamps(),
    # but I need the original `ds` index column, so we do .loc[date+offset:date+offset] to get the slice (which has the
    # index still).
    ty = pd.concat([x.loc[date + offset:date + offset] for offset in [pd.DateOffset(days=-7 * i) for i in lags]])
    ly = x.loc[ty.index + pd.DateOffset(days=-364)]

    return (ty.reset_index(drop=True) / ly.reset_index(drop=True)).mean()


def standardize(x):
    """
    Standardizes a set of data.

    :param x: Arraylike containing the values you wish to standardize.
    :return: Arraylike containing the standardized values.
    """
    return (x - x.mean()) / x.std()


def who_should_i_listen_to():
    """
    Good vibes ;^)

    :return: A good artist to listen to while coding.
    """
    return rand.choice(['Dosbomb', 'City Girl', 'Mystery Skulls', 'GRiZ', 'Caravan Palace', 'Chance the Rapper',
                        'SwuM', 'Busdriver', 'Joey Pecoraro', 'Lemaitre', 'Jar Jar Jr', 'jhfly', 'Potsu',
                        'Zack Villere', 'ProleteR', 'Fox Stevenson', 'Big Wild'])


class PrintAll(object):
    """ A shorthand alias for pd.option_context(). Used to temporarily print the entirety of large dataframes.
        Usage:

        In[1]:  with PrintAll():
                    print(df)  # df is really large/has lots of columns

        **really big dataframe**
    """
    def __init__(self, mrows=None, mcols=None):
        self.init_max_rows = pd.get_option('display.max_rows')
        self.init_max_cols = pd.get_option('display.max_columns')

        self.mrows = mrows
        self.mcols = mcols

    def __enter__(self):
        pd.set_option('display.max_rows', self.mrows)
        pd.set_option('display.max_columns', self.mcols)

    def __exit__(self, *_):
        pd.set_option('display.max_rows', self.init_max_rows)
        pd.set_option('display.max_columns', self.init_max_cols)


class Suppressor(object):
    """
    Suppresses stdout (but not stderr, which can be used as a progress bar, for example, later on).
    Useful for silencing Stan outputs when model fitting in Prophet.
    See https://github.com/facebook/prophet/issues/223

    Usage:
        def fc(x):
            model = Prophet().fit(x)
            forecast = model.predict(model.make_future_dataframe(365)
            return forecast

        finished = []
        work = [timeseries1, timeseries2, timeseries3, timeseries4]

        with Suppressor():
            with mp.Pool(8) as p:
                mp.freeze_support()
                imap_generator = p.imap_unordered(, work, chunksize=25)

                for i, fcst in enumerate(imap_generator):
                    print(f"{round(100 * i/len(work), 2)}% completed.", file=sys.stderr)
                    finished.append(fcst)

    """

    def __init__(self):
        # Open a null file
        self.null_fds = os.open(os.devnull, os.O_RDWR)

        # Save the actual stdout to reassign later
        self.save_fds = os.dup(1)

    def __enter__(self):
        # Assign the null pointers to stdout
        os.dup2(self.null_fds, 1)

    def __exit__(self, *_):
        # Re-assign the real stdout
        os.dup2(self.save_fds, 1)
        # Close the null files
        for fd in [self.null_fds, self.save_fds]:
            os.close(fd)


class Timer(object):
    """A simple context object for timing things when %timeit isn't good enough (ie. multi-line processes). Usage:

        In[1]:  with Timer('Foobar...'):
                    time.sleep(5)

        Foobar...
        Done. Elapsed time: 5.01s

    """
    def __init__(self, msg="Timing..."):
        """

        :param msg: String detailing what process the timer is timing.
        """
        self.msg = msg

    def __enter__(self):
        print(self.msg)
        self.s = time.time()

    def __exit__(self, *_):
        print(f"Done. Elapsed time: {round(time.time() - self.s, 2)}s\n")


class Fcst:
    """Fits, forecasts, and generates diagnostic statistics for a Prophet model."""

    def __init__(self, dat, m, future, acts=None, plot=True, plot_changepoints=True, plot_components=False,
                 boxcox=False, clip_trend=False, round_yhat=True):  # TODO
        """
        Fits and forecasts a given Prophet model. Optional plotting as well as diagnostic information.
        Dates must be in a column/index labeled 'ds', and y-values must be in a column labeled 'y'.

        :param dat: Series containing y-values, with index containing dates, and any conditions for
                    conditional seasonality and/or regressors.
        :param m: Unfitted Prophet() model object.
        :param future: DataFrame containing dates to be forecast and any conditions for conditional seasonality and/or
                       regressors.
        :param acts: Optional Series that contains the actual values for dates in the future DataFrame.
        """

        if dat.index.name != 'ds':
            raise ValueError("Data must include a 'ds' index column.")
        else:
            self.dat = dat

        if future.index.name != 'ds':
            raise ValueError("Future df must include a 'ds' index column.'")
        else:
            self.future = future

        if acts.index.name != 'ds':
            raise ValueError("Acts Series must include a 'ds' index column.")
        elif acts.name != 'y':
            raise ValueError("Acts Series must be named 'y'.")
        else:
            self.acts = acts

        self.m = m
        self.m.fit(self.dat.reset_index())
        self.f = self.m.predict(future.reset_index()).set_index('ds')

        if clip_trend:
            for i in ['', '_lower', '_upper']:
                self.f['trend' + i].loc[self.dat.index[-1]:] = self.f['trend' + i].loc[self.dat.index[-1]]
                self.f['yhat' + i] = self.f['trend' + i] * (1 + self.f['multiplicative_terms' + i]) + self.f['additive_terms' + i]

        if round_yhat:
            self.f['yhat'] = np.round(self.f['yhat'])
        try:
            self.insample_resids = (self.f.loc[self.dat.index]['yhat'] - self.dat['y']).rename('Insample Residuals')
            self.insample_mae = np.abs(self.insample_resids).mean()
            self.insample_rmse = np.sqrt(np.mean(np.square(self.insample_resids)))
            self.insample_mape = (np.abs(self.insample_resids / self.dat['y'])).mean() * 100
            self.insample_maape = (np.arctan(np.abs(self.insample_mae / self.dat['y']))).mean() * 100
        except KeyError:
            print("Future and data have no dates in common. Skipping insample diagnostics.")
            self.insample_resids = None
            self.insample_mae = None
            self.insample_rmse = None
            self.insample_mape = None
            self.insample_maape = None

        if acts is not None:
            self.outsample_resids = (self.f.loc[self.dat.index[-1]:]['yhat'] - self.acts).rename("Outsample Residuals")
            self.outsample_mae = np.abs(self.outsample_resids).mean(skipna=True)
            self.outsample_rmse = np.sqrt(np.mean(np.square(self.outsample_resids)))
            self.outsample_mape = (np.abs(self.outsample_resids / self.acts)).mean() * 100
            self.outsample_maape = (np.arctan(np.abs(self.outsample_mae / self.acts))).mean() * 100
        else:
            self.outsample_resids = None
            self.outsample_mae = None
            self.outsample_mape = None
            self.outsample_rmse = None
            self.outsample_maape = None

        if plot:
            self.plot(plot_changepoints)
            if plot_components:
                self.plot_components()

        print(self)

    def __repr__(self):
        diagnostics = ["Forecast Diagnostics",
                       " ",
                       "In Sample: ",
                       f"\tMAE:   {self.insample_mae}",
                       f"\tRMSE:  {self.insample_rmse}%",
                       f"\tMAPE:  {self.insample_mape}",
                       f"\tMAAPE: {self.insample_maape}%",
                       " ",
                       "Out of Sample: ",
                       f"\tMAE:   {self.outsample_mae}",
                       f"\tRMSE:  {self.outsample_rmse}",
                       f"\tMAPE:  {self.outsample_mape}%",
                       f"\tMAAPE: {self.outsample_maape}%"]

        return "\n| ".join(diagnostics)

    def __str__(self):
        return self.__repr__()

    def diagnostics(self):
        return {'Insample MAE': self.insample_mae, 'Insample RMSE': self.insample_rmse,
                'Insample MAPE': self.insample_mape, 'Insame MAAPE': self.insample_maape,
                'Outsample MAE': self.outsample_mae, 'Outsample RMSE': self.outsample_rmse,
                'Outsample MAPE': self.outsample_mape, 'Outsample MAAPE': self.outsample_maape
                }

    def plot(self, changepoints=True):
        """
        Plots actuals and forecasted values.

        :param changepoints: Bool, whether or not to add changepoints and trendline to plot.

        :return: Matplotlib Figure
        """
        fig = self.m.plot(self.f.reset_index())
        if changepoints:
            add_changepoints_to_plot(fig.gca(), self.m, self.f.reset_index())

        if self.acts is not None:
            self.acts.plot(fig=fig, ls='', marker='D', mec='black', mfc='lightgray', markersize=3)

        return fig

    def plot_components(self):
        """
        Plots the model's seasonal and trend components.

        :return: Matplotlib Figure
        """
        return self.m.plot_components(self.f.reset_index())


