import pandas as pd
import numpy as np
from numpy import intersect1d as inter
from numpy import setdiff1d as diff
from numpy import union1d as union
import ipdb
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
        if not self.col_name in df.columns:
            return df
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
        #ipdb.set_trace()

        if not self.col_name in df.columns:
            return df

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
    # TODO Setup instant drops
    pass
    # drops = ['fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'state',
    #     'fiModelSeries','fiModelDescriptor','ProductGroupDesc',
    #     'fiProductClassDesc','datasource','auctioneerID']
    # df = df.drop(drops, axis=1)
    return df


class Preprocessor:
    def __init__(self, df):


        self.target_cols = np.array(['churn'])

        self.index_cols  = np.array(['SalesID','ModelID','MachineID','ProductGroup',
               'SaleDate','SaleYear', 'YearMade','YearsFrom2012'])
        # self.intercept_col = np.array(['InterpIntercept'])

        self.feature_cols = diff(df.columns.values, union(self.target_cols, self.index_cols))

        #force_numeric_cols = np.array([])
        df = coerce_numerics_(df, force_numeric_cols)

        # organize features into numeric and categorical:
        object_cols = inter ( df.columns[df.dtypes == np.object].values, self.feature_cols )
        self.object_cols = np.array(object_cols)
        self.numeric_cols = diff(self.feature_cols, self.object_cols)

        # set up regularization process:
        # (dictionary of col_name: Regularizer objects)
        self.regularizers = {col: Regularizer(df[col]) for col in self.numeric_cols}

        # set up encoders
        self.encoders = {}
        for col in self.object_cols:
            category_numbers=2
            self.encoders[col] = OneHotEncoder(df,col,category_numbers)


    def coerce_numerics_(self, df, force_numeric_cols):
        to_num = inter(df.columns.values, force_numeric_cols)
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
        #ipdb.set_trace()
        df = self.coerce_numerics_(df)
        df = self.execute_regularization_(df)
        # df = self.execute_encoding_(df)
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
