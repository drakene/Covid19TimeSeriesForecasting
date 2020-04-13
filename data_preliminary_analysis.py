import os
import time
import sys
import pandas as pd
import numpy as np
import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


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


# The idea is that we can build regression models based on different features, or attributes, recorded for each county
# in each state or province.

# TODO:
# Convert date into days since first case
#

def build_regression_models(timeseries_dataset):
    datasets_bystate_index = []  # Used to index data frames stored in datasets_bystate
    datasets_bystate = []
    responses_bystate = []  # Used to keep track of the predictions of each random forest for each state

    for state in timeseries_dataset.Province_State.unique():
        datasets_bystate_index.append(state)
        datasets_bystate.append(timeseries_dataset.loc[timeseries_dataset['Province_State'] == state])

    for state_index in range(len(datasets_bystate)):
        dataset = datasets_bystate[state_index].drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS',
                                                              'Admin2', 'Province_State', 'Country_Region', 'Lat',
                                                              'Long_', 'Combined_Key'])
        timestamp_models = []
        for timestamp in range(3, datasets_bystate[state_index].shape[1]):
            rfr = RandomForestRegressor(random_state=0)

            X, y = dataset.iloc[:, 0:timestamp-1], dataset.iloc[:, timestamp-1]
            if X.shape[0] >= 7:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
            elif X.shape[0] == 1:
                X_train, y_train = X, y
            else:
                X_train, X_test = X.iloc[0:dataset.shape[0] - 1], X.iloc[dataset.shape[0] - 1]
                y_train, y_test = y.iloc[0:dataset.shape[0] - 1], y.iloc[dataset.shape[0] - 1]

            rfr.fit(X_train, y_train)
            timestamp_models.append(rfr)
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
