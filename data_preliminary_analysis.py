import os
import time
import sys
import pandas as pd
import numpy as np
import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# We can play with the constants below to see how far we can accurately forecast the growth rate of counties by state
# CONSTANTS
# ____________________________________________________________

PREDICTOR_INTERVAL = 28  # 4 weeks
FORECASTING_INTERVAL = 14  # 2 weeks
REGRESSION_INTERVAL = PREDICTOR_INTERVAL + FORECASTING_INTERVAL


# ______________________________________________________________

def retrieve_data(timeseries_cases_path, timeseries_deaths_path):
    if time.time() - os.path.getmtime(timeseries_cases_path) > 86400:
        print('Downloading raw data...')

        # Import Johns Hopkins time series data
        os.system('curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                  'csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
                  ' > ' + '"' + timeseries_cases_path + '"')
        os.system('curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                  'csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
                  ' > ' + '"' + timeseries_deaths_path + '"')

        print('Finished downloading raw data.')


# TODO:
# Convert date into days since first case
# We will use RMSE to measure accuracy of the model
# (https://medium.com/making-sense-of-data/time-series-next-value-prediction-using-regression-over-a-rolling-window-228f0acae363
# The idea is that we can build regression models based on different features, or attributes, recorded for each county
# in each state or province.


# Shift the day the cases began in the county to the first column of the data frame

def shift_cases(data_row):
    i = data_row.shape[1]

    while data_row[0, i] != 0:
        i -= 1

    for val in range(i, data_row.shape[1]):
        data_row[0, val - i] = data_row[0, val]

    return data_row


def build_regression_models(timeseries_dataset):
    datasets_bystate_index = []  # Used to index data frames stored in datasets_bystate
    datasets_bystate = []
    regression_model_bystate = []  # Used to keep track random forest models for each state

    # Data Pre-processing

    for state in timeseries_dataset.Province_State.unique():
        datasets_bystate_index.append(state)
        datasets_bystate.append(timeseries_dataset.loc[timeseries_dataset['Province_State'] == state])

    for state_index in range(len(datasets_bystate)):
        state_dataset = datasets_bystate[state_index].drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS',
                                                              'Admin2', 'Province_State', 'Country_Region', 'Lat',
                                                              'Long_', 'Combined_Key'])

        print("Building model for state: " + datasets_bystate_index[state_index])

        # Transforming and cleaning data. Check to see if the prediction and target intervals are available for the
        # county being examined by the for-loop. The interval of time needed to make an accurate prediction is
        # REGRESSION_INTERVAL = PREDICTOR_INTERVAL + FORECASTING_INTERVAL. If the county examined has not had cases
        # for the requisite amount of time specified by REGRESSION_INTERVAL, then it will be dropped from the model.
        # We will test the performance of a recursive multi-step forecasting model built by RandomForestRegressor.

        # We keep track of the models we have built for each time step in the forecasting interval so that we can
        # analyze the performance of each model collectively

        regression_model_bytimestep = []
        if state_dataset.shape[0] < 8:

            for county in range(0, state_dataset.shape[0]):

                lag_count = 0  # Num. of days the county has NOT been infected.
                for date in range(state_dataset.shape[1]):

                    # Here, we are counting zeros and adding them to lag_count to shift lag_count columns so that
                    # the data is in the format of (day 1, day 2, day 3) where each day is a day that the county
                    # is infected with the virus.
                    if state_dataset.loc[county, date] == 0:
                        lag_count += 1
                    elif lag_count - state_dataset.shape[1] < REGRESSION_INTERVAL:
                        state_dataset.drop(state_dataset.index[county])
                        print("Unable to build model for state:" + datasets_bystate_index[
                            state_index]) + " due to insufficient" \
                                            "data."
                        break
                    else:
                        shifted_row = shift_cases(state_dataset.loc[county])
                        state_dataset.loc[county] == shifted_row.loc[0]
                        break




        else:
            print("Unable to build model for state:" + datasets_bystate_index[state_index]) + " due to insufficient" \
                                                                                              "data."

    responses_bystate[state_index] = timestamp_models

    return responses_bystate

    # print(dataset_time_series)


def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    # Dataset paths
    timeseries_cases_path = os.path.join('Covid-19 Data', 'time_series_covid19_confirmed_US.csv')
    timeseries_deaths_path = os.path.join('Covid-19 Data', 'time_series_covid19_deaths_US.csv')

    # Retrieve data from GitHub of Johns Hopkins and New York Times
    retrieve_data(timeseries_cases_path, timeseries_deaths_path)

    # Making data frames from csv's
    timeseries_cases_dataset = pd.read_csv(timeseries_cases_path)
    timeseries_deaths_dataset = pd.read_csv(timeseries_deaths_path)

    print(timeseries_cases_dataset.groupby('Province_State').nunique())

    # build_regression_models(timeseries_cases_dataset)


main()
