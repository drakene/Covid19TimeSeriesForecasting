import os
import time
import sys
import pandas as pd
import numpy as np
import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# We can play with the constants below to see how far we can accurately forecast the growth rate of counties by state
# CONSTANTS
# ____________________________________________________________

PREDICTION_INTERVAL = 28  # 4 weeks
TARGET_INTERVAL = 14  # 2 weeks
REGRESSION_INTERVAL = PREDICTION_INTERVAL + TARGET_INTERVAL

#______________________________________________________________

def retrieve_data(county_level_path, state_level_path, timeseries_cases_path, timeseries_deaths_path):
    if time.time() - os.path.getmtime(county_level_path) > 86400:
        print('Downloading raw data...')

        # Import New York Times data on states and counties
        os.system('curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
                  ' > ' + '"' + county_level_path + '"')
        os.system('curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
                  ' > ' + '"' + state_level_path + '"')

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
# We will use MAPE, if we can, to measure success
# (https://medium.com/making-sense-of-data/time-series-next-value-prediction-using-regression-over-a-rolling-window-228f0acae363
# The idea is that we can build regression models based on different features, or attributes, recorded for each county
# in each state or province.


def build_regression_models(timeseries_dataset):
    datasets_bystate_index = []  # Used to index data frames stored in datasets_bystate
    datasets_bystate = []
    responses_bystate = []  # Used to keep track random forest models for each state

    # Data Pre-processing

    for state in timeseries_dataset.Province_State.unique():
        datasets_bystate_index.append(state)
        datasets_bystate.append(timeseries_dataset.loc[timeseries_dataset['Province_State'] == state])

    for state_index in range(len(datasets_bystate)):
        dataset = datasets_bystate[state_index].drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS',
                                                              'Admin2', 'Province_State', 'Country_Region', 'Lat',
                                                              'Long_', 'Combined_Key'])

        print("Building model for state: " + datasets_bystate_index[state_index])

        # Transforming and cleaning data. Check to see if the prediction and target intervals are available for the
        # county being examined by the for-loop. The interval of time needed to make an accurate prediction is
        # REGRESSION_INTERVAL = PREDICTION_INTERVAL + TARGET_INTERVAL. If the county examined has not had cases
        # for the requisite amount of time specified by REGRESSION_INTERVAL, then it will be dropped from the model.
        # We will test the performance of a range of rolling windows in days (1, PREDICTION_INTERVAL) and choose the
        # model with the best performance

        regression_models_bystate = []
        for county in range(0, datasets_bystate[state_index].shape[0]):
            regression_models_byrollingwindow = []





        responses_bystate[state_index] = timestamp_models

    return responses_bystate

    # print(dataset_time_series)


def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    # Dataset paths
    county_level_path = os.path.join('Covid-19 Data', 'us-counties.csv')
    state_level_path = os.path.join('Covid-19 Data', 'us-states.csv')
    timeseries_cases_path = os.path.join('Covid-19 Data', 'time_series_covid19_confirmed_US.csv')
    timeseries_deaths_path = os.path.join('Covid-19 Data', 'time_series_covid19_deaths_US.csv')

    # Retrieve data from GitHub of Johns Hopkins and New York Times
    retrieve_data(county_level_path, state_level_path, timeseries_cases_path, timeseries_deaths_path)

    # Making data frames from csv's
    county_level_dataset = pd.read_csv(county_level_path)
    state_level_dataset = pd.read_csv(state_level_path)
    timeseries_cases_dataset = pd.read_csv(timeseries_cases_path)
    timeseries_deaths_dataset = pd.read_csv(timeseries_deaths_path)

    print(timeseries_cases_dataset.groupby('Province_State').nunique())
    build_regression_models(timeseries_cases_dataset)


main()
