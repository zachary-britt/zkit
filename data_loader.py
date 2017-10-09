import pandas as pd
import pickle
import pdb

def data_loader(run_cached_df = False, proportion=1.0):

    if run_cached_df:
        # get cached dataframes/ read from pickle
        df_train = pd.read_pickle('cache/df_train.pkl')
        df_test = pd.read_pickle('cache/df_test.pkl')

    else:
        if proportion < 1.0: #skip rows to reduce df size
            # nth row to keep
            n = int(1.0 / proportion)
            # length of dataset
            row_count = sum(1 for row in open('data/train.csv'))
            # Row indices to skip
            skipped = [x for x in range(1, row_count) if x % n != 0]
        else:
            skipped = None

        #read from csv
        nulls = 'None or Unspecified'
        df_train = pd.read_csv('data/train.csv', skiprows=skipped, na_values = nulls )
        df_test = pd.read_csv('data/test.csv', skiprows=skipped, na_values = nulls)

        #convert saledate to sale_year
        # (takes forever, so do it here to cache result)
        df_train['SaleDate'] = pd.DatetimeIndex(df_train['saledate'])
        df_test['SaleDate'] = pd.DatetimeIndex(df_test['saledate'])

        # pdb.set_trace()

        df_train['SaleYear'] = df_train['SaleDate'].apply(lambda x: x.year)
        df_test['SaleYear'] = df_test['SaleDate'].apply(lambda x: x.year)


        # df_train = df_train.drop('saledate', axis=1)
        # df_test = df_test.drop('saledate', axis=1)

        #cache dataframes
        df_train.to_pickle('cache/df_train.pkl')
        df_test.to_pickle('cache/df_test.pkl')

    return df_train, df_test
