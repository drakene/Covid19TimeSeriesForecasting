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


PREDICTOR_INTERVAL = 14  # 2 weeks
FORECASTING_INTERVAL = 7  # 1 week
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


def build_regression_models(timeseries_dataset):
    datasets_bystate_index = []  # Used to index data frames stored in datasets_bystate
    datasets_bystate = []
    regression_model_bystate = []  # Used to keep track random forest models for each state

    # Transforming and cleaning data. Check to see if the prediction and target intervals are available for the
    # county being examined by the for-loop. The interval of time needed to make an accurate prediction is
    # REGRESSION_INTERVAL = PREDICTOR_INTERVAL + FORECASTING_INTERVAL. If the county examined has not had cases
    # for the requisite amount of time specified by REGRESSION_INTERVAL, then it will be dropped from the model.

    for state in timeseries_dataset.Province_State.unique():
        print("Building model for state: " + state)

        # Set index and labels
        state_dataset = timeseries_dataset.loc[timeseries_dataset['Province_State'] == state].copy()
        state_dataset.index = pd.RangeIndex(len(state_dataset.index))  # Reset index for proper func. of iloc

        # Drop unneeded attributes
        state_dataset.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS',
                                    'Admin2', 'Province_State', 'Country_Region', 'Lat',
                                    'Long_', 'Combined_Key'], inplace=True)
        column_names = []  # Store new column names
        for col in range(1, state_dataset.shape[1] + 1):
            column_names.append("Day " + str(col))
        state_dataset.columns = column_names

        if state_dataset.shape[0] < 6:
            print("Unable to build model for state:" + state + " due to an insufficient number counties.")

        else:
            county = 0
            while county < state_dataset.shape[0]:
                has_dropped = False  # Control variable

                # Here, we are counting zeros and adding them to lag_count to shift lag_count columns so that
                # the data is in the format of (day 1, day 2, day 3) where each day is a day that the county
                # is infected with the virus.

                lag_count = 0  # Num. of days the county has NOT been infected.
                for date in range(state_dataset.shape[1]):
                    if state_dataset.iloc[county, date] == 0:
                        lag_count += 1
                    elif state_dataset.shape[1] - lag_count < REGRESSION_INTERVAL:
                        state_dataset.drop(state_dataset.index[county], inplace=True)
                        state_dataset.index = pd.RangeIndex(len(state_dataset.index))
                        has_dropped = True
                        break
                    elif state_dataset.iloc[county, 0] == 0:
                        # Shift the day the cases began in the county to the first column of the data frame
                        for val in range(lag_count, state_dataset.shape[1]):
                            state_dataset.iloc[county, val - lag_count] = state_dataset.iloc[county, val]
                            state_dataset.iloc[county, val] = 0  # To replace shifted values with 0
                        break
                    # Check to see if we have a county full of no cases... for some reason these are in here!
                    if date == state_dataset.shape[1] - 1 and state_dataset.iloc[county].sum() == 0:
                        state_dataset.drop(state_dataset.index[county], inplace=True)
                        state_dataset.index = pd.RangeIndex(len(state_dataset.index))
                        has_dropped = True

                if not has_dropped:
                    county += 1

                if state_dataset.shape[0] < 6:
                    break

        if state_dataset.shape[0] >= 6:
            state_dataset.drop(state_dataset.iloc[:, -(state_dataset.shape[1] - REGRESSION_INTERVAL):], axis='columns',
                               inplace=True)
            datasets_bystate.append(state_dataset)
            datasets_bystate_index.append(state)

    # Building regression model via multistep recursive forecasting with RandomForestRegressor. We analyze the
    # performance of each model with RMSE.
    for state_dataset in datasets_bystate:
        train, test = train_test_split(state_dataset, test_size=.2)



    return datasets_bystate_index, datasets_bystate


def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Dataset paths
    timeseries_cases_path = os.path.join('Covid-19 Data', 'time_series_covid19_confirmed_US.csv')
    timeseries_deaths_path = os.path.join('Covid-19 Data', 'time_series_covid19_deaths_US.csv')

    # Retrieve data from GitHub of Johns Hopkins
    retrieve_data(timeseries_cases_path, timeseries_deaths_path)

    # Making data frames from csv's
    timeseries_cases_dataset = pd.read_csv(timeseries_cases_path)

    # print(timeseries_cases_dataset.groupby('Province_State').nunique())

    datasets_bystate_index, datasets_bystate = build_regression_models(timeseries_cases_dataset)

    current_removed_state_index = 0
    for i in range(0, len(datasets_bystate)):
        print(datasets_bystate_index[i])
        print(datasets_bystate[i])


main()
