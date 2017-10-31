import pandas as pd
import numpy as np
from numpy import setdiff1d as diff
import pickle
import ipdb
from preprocessor import make_immediate_drops

def data_loader(have_df, proportion=1.0):


    example=True
    # if example:
    #     df_total = pd.read_csv('data/churn.csv')
    #     total_samples = df_total[df_total.columns[0]].values.shape[0]
    #     all_inds = np.arange(total_samples)
    #     test_inds = np.random.choice(all_inds, int(total_samples/10), replace=False)
    #     df_test = df_total.loc[test_inds]
    #     df_train = df_total.loc[ diff(all_inds, test_inds) ]
    #

    if have_df:
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
        # nulls = 'None or Unspecified'
        df_train = pd.read_csv('data/train.csv', skiprows=skipped) #, na_values = nulls )
        df_test = pd.read_csv('data/test.csv', skiprows=skipped) #, na_values = nulls)


        # #convert saledate to sale_year
        # # (takes forever, so do it here to cache result)
        # df_train['SaleDate'] = pd.DatetimeIndex(df_train['saledate'])
        # df_test['SaleDate'] = pd.DatetimeIndex(df_test['saledate'])
        # df_train = df_train.drop('saledate', axis=1)
        # df_test = df_test.drop('saledate', axis=1)
        # # ipdb.set_trace()
        #
        # df_train['SaleYear'] = df_train['SaleDate'].apply(lambda x: x.year)
        # df_test['SaleYear'] = df_test['SaleDate'].apply(lambda x: x.year)





        #make early drops before caching
        df_train = make_immediate_drops(df_train)
        df_test = make_immediate_drops(df_train)


        #cache dataframes
        df_train.to_pickle('cache/df_train.pkl')
        df_test.to_pickle('cache/df_test.pkl')

    return df_train, df_test

def feature_saver(feat_list):
    with open('cache/feats.pkl','wb') as f: pickle.dump(f,feat_list)

def feature_loader():
    with open('cache/feats.pkl','rb') as f: feats = pickle.load(f)
    return feats
