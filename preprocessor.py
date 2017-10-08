import pandas as pd
import numpy as np
from numpy import intersect1d as inter
from numpy import setdiff1d as diff
from numpy import union1d as union
import pdb
import datetime


class Regularizer:
    def __init__(self, series):
        # record mean and variance of training series
        self.mean = series.mean()
        self.var = series.var()

    def __call__(self, series):
        # execute regularization and nan filling
        series = (series - self.mean)/self.var**0.5
        series = series.fillna(0)
        return series

class OneHotEncoder:
    def __init__(self, df, col_name, max_cats=5):
        self.col_name = col_name
        # record dominant categories
        all_cats = df[col_name].value_counts().index.values
        n = len(all_cats)
        if n <= max_cats:
            self.cats = all_cats[:n]
        else:
            self.cats = all_cats[:max_cats]

        self.new_col_names = []
        for cat in self.cats:
            self.new_col_names.append(col_name + "=" + str(cat))

    def __call__(self, df):
        # execute encoding for each category in self.cats
        #pdb.set_trace()
        for i,cat in enumerate(self.cats):
            new_col = self.new_col_names[i]

            #pandas gets upset when a column is all nan
            if df[self.col_name].isnull().all():
                df[new_col] = 0
            else:
                df[new_col] = (df[self.col_name] == cat).astype(int)

        # drop old column
        df = df.drop(self.col_name,axis=1)
        return df


def make_immediate_drops(df):
    drops = ['fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'state',
        'MachineID', 'fiModelSeries','fiModelDescriptor','ProductGroupDesc',
        'SalesID','fiProductClassDesc','datasource','auctioneerID']
    df = df.drop(drops, axis=1)
    return df


def add_age_col(df):
    # nanify nonsense
    df.YearMade = df.YearMade.where(df.YearMade >= 1800, np.nan)
    df['Decade'] = ((df['SaleYear']-1900)/10).astype(int)
    df['Age'] = df['SaleYear'] - df['YearMade']
    return df


class Preprocessor:
    def __init__(self, df):


        # remember nan status of training cols:
        # self.has_nulls = df.columns[df.isnull().any()].values
        # self.all_nulls = df.columns[df.isnull().all()].values

        self.target_col = np.array(['SalePrice'])
        self.feature_cols = diff( df.columns.values, self.target_col)

        # organize columns into categorical and numeric:
        force_categorical = np.array(['ModelID'])
        object_cols = inter ( df.columns[df.dtypes == np.object].values, self.feature_cols )
        self.object_cols = np.concatenate([object_cols, force_categorical], axis=0)
        self.numeric_cols = diff(self.feature_cols, self.object_cols)



        # set up regularization process:
        # (dictionary of col_name: Regularizer objects)
        self.regularizers = {col: Regularizer(df[col]) for col in self.numeric_cols}

        self.encoders = {}
        for col in self.object_cols:
            if col == "ModelID":
                self.encoders[col] = OneHotEncoder(df,col,500)
            else:
                self.encoders[col] = OneHotEncoder(df,col,6)



    def purge_useless_cols_(self, df):
        to_drop = inter(df.columns.values, self.all_nulls)
        df = df.drop(to_drop, axis=1)
        return df

    def purge_every_null_(self, df):
        to_drop = inter(df.columns.values, self.any_nulls)
        df = df.drop(to_drop, axis=1)
        return df

    def coerce_numerics_(self, df):
        to_num = inter(df.columns.values, self.numeric_cols)
        for col in to_num:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def execute_regularization_(self, df):
        for col in self.regularizers:
            df[col] = self.regularizers[col](df[col])
        return df

    def execute_encoding_(self, df):
        for col in self.encoders:
            df = self.encoders[col](df)
        return df

    def __call__(self, df):
        #pdb.set_trace()
        df = self.coerce_numerics_(df)
        df = self.execute_regularization_(df)
        df = self.execute_encoding_(df)
        return df


if __name__ == "__main__":

    df_train = pd.read_csv('data/example_train.csv')
    df_test = pd.read_csv('data/example_test.csv')

    print(df_train)
    print(df_test)

    cleaner = DataClenser(df_train)

    df_train = cleaner(df_train)
    df_test = cleaner(df_test)

    print(df_train)
    print(df_test)





#buffer
