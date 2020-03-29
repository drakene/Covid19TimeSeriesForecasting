import pandas as pd
import sklearn.pipeline
import sklearn.model_selection


def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    dataset_open_line = pd.read_csv('D:\\School Work\\CSCI 334\\Project\\Covid-19 Data\\COVID19_line_list_data.csv')
    dataset_line_list = pd.read_csv('D:\\School Work\\CSCI 334\\Project\\Covid-19 Data\\COVID19_open_line_list.csv')

    print(dataset_line_list.columns)
    print(dataset_open_line.columns)



main()
