import pandas as pd
import os
import sys

import sklearn.pipeline
import sklearn.model_selection


def main():


    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    # Import New York Times data on states and counties
    os.system('curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
              ' > ' + '"' + os.path.join('Covid-19 Data', 'us-counties.csv') + '"')
    os.system('curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
              ' > ' + '"' + os.path.join('Covid-19 Data', 'us-states.csv') + '"')

    # Import Johns Hopkins time series data
    os.system('curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
              'csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
              ' > ' + '"' + os.path.join('Covid-19 Data', 'time_series_covid19_confirmed_US.csv') + '"')
    os.system('curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
              'csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
              ' > ' + '"' + os.path.join('Covid-19 Data', 'time_series_covid19_deaths_US.csv') + '"')

    county_level_path = os.path.join('Covid-19 Data', 'us-counties.csv')
    state_level_path = os.path.join('Covid-19 Data', 'us-states.csv')

    county_level_dataset = pd.read_csv(county_level_path)
    state_level_dataset = pd.read_csv(state_level_path)

    print(county_level_dataset)
    print(state_level_dataset)


main()
