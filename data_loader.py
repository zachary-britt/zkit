import pandas as pd
import pickle


def data_loader(run_cached_df = False, proportion=1.0):

    if run_cached_df:
        # get cached dataframes/ read from pickle
        df_train = pd.read_pickle('cache/df_train.pkl')
        df_test = pd.read_pickle('cache/df_test.pkl')

    else:
        if proportion != 1.0: #skip rows to reduce df size
            # nth row to keep
            n = int(1.0 / proportion)
            # length of dataset
            row_count = sum(1 for row in open(filename))
            # Row indices to skip
            skipped = [x for x in range(1, num_lines) if x % n != 0]
        else:
            skipped = None

        #read from csv
        nulls = 'None or Unspecified'
        df_train = pd.read_csv('data/train.csv', skiprows=skipped, na_values = nulls )
        df_test = pd.read_csv('data/test.csv', skiprows=skipped, na_values = nulls)

        #convert date times
        date_time_cols = ['saledate']
        for col in date_time_cols:
            df_train[col] = pd.DatetimeIndex(df_train[col])
            df_test[col] = pd.DatetimeIndex(df_test[col])

        #cache dataframes
        df_train.to_pickle('cache/df_train.pkl')
        df_test.to_pickle('cache/df_test.pkl')

    return df_train, df_test
