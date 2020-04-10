import pandas as pd
import os
import time
import sys

import sklearn.pipeline
import sklearn.model_selection


def main():

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    # Data sets
    county_level_path = os.path.join('Covid-19 Data', 'us-counties.csv')
    state_level_path = os.path.join('Covid-19 Data', 'us-states.csv')
    timeseries_cases = os.path.join('Covid-19 Data', 'time_series_covid19_confirmed_US.csv')
    timeseries_deaths = os.path.join('Covid-19 Data', 'time_series_covid19_deaths_US.csv')

    if os.path.getmtime(county_level_path) - time.time() > 86400:

        print('Downloading raw data...')

        # Import New York Times data on states and counties
        os.system('curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
                  ' > ' + '"' + county_level_path + '"')
        os.system('curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
                  ' > ' + '"' + state_level_path + '"')

        # Import Johns Hopkins time series data
        os.system('curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                  'csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
                  ' > ' + '"' + timeseries_cases + '"')
        os.system('curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                  'csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
                  ' > ' + '"' + timeseries_deaths + '"')

        print('Finished downloading raw data.')

    county_level_dataset = pd.read_csv(county_level_path)
    state_level_dataset = pd.read_csv(state_level_path)
    print(county_level_dataset)
    print(state_level_dataset)

    print('______________________________STATE OF WYOMING______________________________')
    print(county_level_dataset.loc[(county_level_dataset['state'] == 'Wyoming') &
                                   (county_level_dataset['county'] == 'Sheridan')])


main()
