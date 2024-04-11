import pandas as pd
import numpy as np
import time

import hvplot
import holoviews as hv
import hvplot.dask

import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
from dask.distributed import Client

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.linear_model import PoissonRegressor  # for Poisson regression
from sklearn.preprocessing import PolynomialFeatures  # for polynomial features
from sklearn.linear_model import LogisticRegression  # for logistic regression
from sklearn.linear_model import LinearRegression  # for linear regression
# i imported them all so i can switch between them and see

from sklearn.metrics import mean_squared_error
# apparently this helps us during regression

def pandas_test():
    df1 = pd.read_csv('trips_by_distance.csv')  # import one csv file, this one's big
    df2 = pd.read_csv('trips_full_data.csv')  # import the other csv file which is smaller

    # print(df1)
    # print(df2)

    # How many people are staying at home?

    # print(df1.groupby(by='Week')['Population Staying at Home'].mean())

    dfa1 = df1.dropna()  # clean nulls
    dfa1.drop_duplicates(subset=['Row ID'], keep='first', inplace=True)  # clean duplicates
    dfa1 = dfa1.groupby(by='Population Staying at Home')['Week'].mean().hist(
        bins=44)  # turn into mean average dataframe then make it a histogram
    # self-explanatory beyond this
    dfa1.set_title("How many people are staying at home per week?")
    dfa1.set_xlabel("Week")
    dfa1.set_ylabel("Population Staying at Home")
    plt.savefig('histogram_a1.png')
    plt.clf()  # clear matplotlib cache or whatever

    # How far are people travelling when they don't stay home?

    dfa2 = df2[['Trips 1-25 Miles', 'Trips 1-3 Miles', 'Trips 10-25 Miles',
                'Trips 100-250 Miles', 'Trips 100+ Miles', 'Trips 25-100 Miles',
                'Trips 25-50 Miles', 'Trips 250-500 Miles', 'Trips 250-500 Miles',
                'Trips 3-5 Miles', 'Trips 5-10 Miles',
                'Trips 50-100 Miles', 'Trips 500+ Miles']]  # select these rows only
    dfa2 = dfa2.mean()  # average them out

    xvalues = ['Trips 1-25 Miles', 'Trips 1-3 Miles', 'Trips 10-25 Miles',
               'Trips 100-250 Miles', 'Trips 100+ Miles', 'Trips 25-100 Miles',
               'Trips 25-50 Miles', 'Trips 250-500 Miles', 'Trips 250-500 Miles',
               'Trips 3-5 Miles', 'Trips 5-10 Miles',
               'Trips 50-100 Miles', 'Trips 500+ Miles']  # to help the graph later
    yvalues = dfa2.tolist()  # convert dataframe values to list for graph

    # the histogram gets plotted
    plt.figure(figsize=(10, 6))  # good for controlling the size of our bars
    plt.bar(xvalues, yvalues)  # bars get actual x value and y value from before
    plt.xlabel('Trip Distance Range')  # x-axis label
    plt.ylabel('Average Number of Trips')  # y-axis label
    plt.title('Distribution of Trips by Distance Range')  # plot title, what this graph is talking about
    plt.yscale(
        'log')  # logarithmic scaling to y-axis for better visualization of large values, it looks a lot more beautiful than linear scaling in testing

    # another histogram gets plotted
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels so you can read them better
    plt.tight_layout()  # it makes it look better, but i don't quite know why though
    plt.savefig('histogram_a2.png')  # save it
    plt.clf()  # clear matplotlib cache or whatever

    # The dates that > 10000000 people conducted 10-25 number of trips

    dfb1 = df1[df1['Number of Trips 10-25'] > 10000000]  # i only will want more than 10,000,000 in the values
    dfb1 = dfb1.reset_index(drop=True)  # reset the new index
    dfb1.plot(kind='scatter', x='Date', y='Number of Trips 10-25')  # scatter plot for date against frequency of this
    plt.title("Number of Trips 10-25 across time")  # correct title
    plt.xlabel("Date")  # correct label
    plt.gca().xaxis.set_major_locator(
        ticker.AutoLocator())  # if i don't do this, the labels for date make up a solid black blob
    plt.ylim(100000000, 260000000)  # y limit is helpful otherwise you will see rubbish in the plot
    plt.savefig('scatterplot_b1.png')  # save
    plt.clf()  # clear matplotlib cache or whatever

    dfb2 = df1[df1['Number of Trips 50-100'] > 10000000]  # i only will want more than 10,000,000 in the values
    dfb2 = dfb2.reset_index(drop=True)  # reset the new index
    dfb1.plot(kind='scatter', x='Date', y='Number of Trips 50-100')  # scatter plot for date against frequency
    plt.title("Number of Trips 50-100 across time")  # correct title
    plt.xlabel("Date")  # correct label
    plt.gca().xaxis.set_major_locator(
        ticker.AutoLocator())  # if i don't do this, the labels for date make up a solid black blob
    plt.ylim(10000000, 30000000)  # y limit is helpful otherwise you will see rubbish in the plot
    plt.savefig('scatterplot_b2.png')  # save
    plt.clf()  # clear matplotlib cache or whatever


def dask_test(schedulerIn, processors):
    with ((dask.config.set(scheduler=schedulerIn, n_workers=processors))):
        start_time_dask = time.time()  # so we can know how long this dask instance takes
        dfd1a1 = dd.read_csv('trips_by_distance.csv', blocksize=16e6, dtype={'Population Staying at Home': 'float64',
                                                                             'Week': 'int64',
                                                                             'County Name': 'object',
                                                                             'Number of Trips': 'float64',
                                                                             'Number of Trips 1-3': 'float64',
                                                                             'Number of Trips 10-25': 'float64',
                                                                             'Number of Trips 100-250': 'float64',
                                                                             'Number of Trips 25-50': 'float64',
                                                                             'Number of Trips 250-500': 'float64',
                                                                             'Number of Trips 3-5': 'float64',
                                                                             'Number of Trips 5-10': 'float64',
                                                                             'Number of Trips 50-100': 'float64',
                                                                             'Number of Trips <1': 'float64',
                                                                             'Number of Trips >=500': 'float64',
                                                                             'Population Not Staying at Home': 'float64',
                                                                             'State Postal Code': 'object'
                                                                             })  # 16 megabyte chunks dataframe
        dfd1a1 = dfd1a1.dropna()  # remove nulls
        dfd1a1 = dfd1a1[['Population Staying at Home', 'Week']]  # select only these rows
        dfd1a1 = dfd1a1.groupby(by='Population Staying at Home')['Week'].mean()  # average it out

        dfd1b1 = dd.read_csv('trips_by_distance.csv', blocksize=16e6, dtype={'Population Staying at Home': 'float64',
                                                                             'Week': 'int64',
                                                                             'County Name': 'object',
                                                                             'Number of Trips': 'float64',
                                                                             'Number of Trips 1-3': 'float64',
                                                                             'Number of Trips 10-25': 'float64',
                                                                             'Number of Trips 100-250': 'float64',
                                                                             'Number of Trips 25-50': 'float64',
                                                                             'Number of Trips 250-500': 'float64',
                                                                             'Number of Trips 3-5': 'float64',
                                                                             'Number of Trips 5-10': 'float64',
                                                                             'Number of Trips 50-100': 'float64',
                                                                             'Number of Trips <1': 'float64',
                                                                             'Number of Trips >=500': 'float64',
                                                                             'Population Not Staying at Home': 'float64',
                                                                             'State Postal Code': 'object'
                                                                             })  # just grab the same file again, but not null like the last one

        dfd2 = dd.read_csv('trips_full_data.csv', blocksize=20e6)  # the tiny file is only 300 KB, 20e6 is for nothing
        dfd2a1 = dfd2[['Trips 1-25 Miles', 'Trips 1-3 Miles', 'Trips 10-25 Miles',
                       'Trips 100-250 Miles', 'Trips 100+ Miles', 'Trips 25-100 Miles',
                       'Trips 25-50 Miles', 'Trips 250-500 Miles', 'Trips 250-500 Miles',
                       'Trips 3-5 Miles', 'Trips 5-10 Miles',
                       'Trips 50-100 Miles', 'Trips 500+ Miles']].mean()  # just these columns then average them

        dfd1a2, dfd1b2, dfd2a2 = dask.compute(dfd1a1, dfd1b1, dfd2a1)  # compute them all parallel at the same time

        dfda1 = dfd1a2.to_frame()  # import as a dataframe not series
        dfda1.plot.hist(bins=44, column=[
            "Week"])  # the column flag is there for the histogram because this wouldn't work otherwise
        plt.title("How many people are staying at home per week?")  # correct title
        plt.xlabel("Week")  # correct x axis label
        plt.ylabel("Population Staying at Home")  # correct y axis label
        plt.savefig('histogram_a1d.png')  # save it
        plt.clf()  # clear matplotlib cache or whatever

        dfda2 = dfd2a2  # grab the output of previous dask compute
        xvalues = ['Trips 1-25 Miles', 'Trips 1-3 Miles', 'Trips 10-25 Miles',
                   'Trips 100-250 Miles', 'Trips 100+ Miles', 'Trips 25-100 Miles',
                   'Trips 25-50 Miles', 'Trips 250-500 Miles', 'Trips 250-500 Miles',
                   'Trips 3-5 Miles', 'Trips 5-10 Miles',
                   'Trips 50-100 Miles', 'Trips 500+ Miles']  # setting up our bar chart
        yvalues = dfda2  # reassign name to yvalues for readability in the following functions
        plt.figure(figsize=(10, 6))  # helps it look neat
        plt.bar(xvalues, yvalues)  # bars out of values, a bar for each x value, and it has y value height
        plt.xlabel('Trip Distance Range')  # correct x axis label
        plt.ylabel('Average Number of Trips')  # correct y axis label
        plt.title('Distribution of Trips by Distance Range')  # correct title
        plt.yscale('log')  # logarithmic scale actually makes sense here

        plt.xticks(rotation=45, ha='right')  # rotate x-axis labels so they can be read easier
        plt.tight_layout()  # looks nicer
        plt.savefig('histogram_a2d.png')  # save it
        plt.clf()  # clear matplotlib cache or whatever

        dfdb1 = dfd1b2[dfd1b2['Number of Trips 10-25'] > 10000000]  # just want the 10,000,000 or above data points
        dfdb1 = dfdb1.reset_index(drop=True)  # reset our new df indices
        dfdb1.plot(kind='scatter', x='Date', y='Number of Trips 10-25')  # scatter plot
        plt.title("Number of Trips 10-25 across time")  # correct title
        plt.xlabel("Date")  # correct label
        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())  # the date ticks become so much spam otherwise
        plt.ylim(100000000, 260000000)  # get rid of the rubbish at the bottom
        plt.savefig('scatterplot_b1d.png')  # save
        plt.clf()  # clear matplotlib cache or whatever

        dfdb2 = dfd1b2[dfd1b2['Number of Trips 50-100'] > 10000000]  # just want the 10,000,000 or above data points
        dfdb2 = dfdb2.reset_index(drop=True)  # reset our new df indices
        dfdb2.plot(kind='scatter', x='Date', y='Number of Trips 50-100')  # scatter plot
        plt.title("Number of Trips 50-100 across time")  # correct title
        plt.xlabel("Date")  # correct label
        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())  # the date ticks become so much spam otherwise
        plt.savefig('scatterplot_b2d.png')  # save
        plt.clf()  # clear matplotlib cache or whatever

        dask_time = time.time() - start_time_dask  # we finished, save the time, that's important
        return dask_time  # send it back so we can map the cpu time


if __name__ == "__main__":  # quirky programmer convention where you can import this file without running everything in it

    pd.set_option('display.max_columns',
                  None)  # this and the line below enabled debugging, without it i'd get the ... which was super annoying.
    pd.set_option('display.max_rows', None)
    n_processors = [('single-threaded', 10), ('threads', 10), (
        'threads', 20)]  # behold my TEST CASES for dask. sequential, parallel with 10, and parallel with 20 processors.
    n_processors_time = {}  # for later of course.
    start_time_pandas = time.time()  # just pandas benchmark
    pandas_test()  # run the pandas version of this program
    pandas_time = time.time() - start_time_pandas  # end benchmark
    n_processors_time['pandas-single-threaded', 0] = pandas_time  # store the first test case.

    for scheduler, processor in n_processors:  # try everything
        n_processors_time[scheduler, processor] = dask_test(scheduler, processor)  # dask benchmark

    print(n_processors_time)  # debug for time, show actual performance across all these.

    # cpu time chart
    plotOut = pd.DataFrame(n_processors_time, index=[0]).plot(kind='bar',
                                                              grid=True)  # pass the dict as if it were a dataframe
    plotOut.legend(
        ["Pandas Sequential", "Dask Sequential", "Dask Parallel 10 processors", "Dask Parallel 20 processors"],
        loc='lower left')  # so the graph makes sense to a human
    plt.xlabel("Processing Method")  # correct x label
    plt.ylabel("Time taken")  # correct y label
    plt.title("Different speeds for different methods.")  # correct title
    # plt.xticks()  # i was gonna use this but i forgot how and didn't bother
    plt.savefig('cpu_time.png')  # save it

    # MODEL TIME

    # people in relation to distance imported with pandas, nulls erased, relevant columns extracted and summated.
    data1 = pd.read_csv("trips_by_distance.csv")
    data1.dropna(inplace=True)
    data1 = data1[['Number of Trips <1', 'Number of Trips 1-3', 'Number of Trips 3-5', 'Number of Trips 5-10',
                   'Number of Trips 10-25', 'Number of Trips 25-50', 'Number of Trips 50-100',
                   'Number of Trips 100-250',
                   'Number of Trips 250-500', 'Number of Trips >=500']].sum()

    # people in relation to trips imported with pandas, nulls erased, relevant columns extracted and summated.
    data2 = pd.read_csv("trips_full_data.csv")
    data2.dropna(inplace=True)
    data2 = data2[['Trips 1-25 Miles', 'Trips 1-3 Miles', 'Trips 10-25 Miles', 'Trips 100-250 Miles',
                   'Trips 25-50 Miles', 'Trips 250-500 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles',
                   'Trips 50-100 Miles', 'Trips <1 Mile', 'Trips 500+ Miles']].sum()

    y = data1.values
    X = data2[:-1].values.reshape(-1, 1)  # if i leave the last value in then a mystical creature called ValueError will appear.

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # seperate data for training. commented out because useless.

    model = LinearRegression()  # our model
    model.fit(X, y)  # fitting our data to the model
    r_sq = model.score(X, y)  # the r^2 score for evaluation
    y_pred = model.predict(X)  # predict the response
    mse = mean_squared_error(y, y_pred)  # mean squared error for evaluation

    print(f"coefficient of determination: {r_sq}")
    print(f"mean square error: {mse}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")
    print(f"predicted response:\n{y_pred}")

    # finished!!!!
