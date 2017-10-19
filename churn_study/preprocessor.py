import pandas as pd
import numpy as np
from numpy import intersect1d as inter
from numpy import setdiff1d as diff
from numpy import union1d as union
from functools import reduce
import ipdb
import datetime
from sklearn.linear_model import Ridge



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
    drops = ['fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'state',
        'fiModelSeries','fiModelDescriptor','ProductGroupDesc',
        'fiProductClassDesc','datasource','auctioneerID']
    df = df.drop(drops, axis=1)
    return df


def add_age_col(df):
    # nanify nonsense
    df.YearMade = df.YearMade.where(df.YearMade >= 1800, np.nan)

    df['Age'] = df['SaleYear'] - df['YearMade']
    df['YearsFrom2012'] = df['SaleYear'] - 2012
    return df

class PriceScale:
    def __init__(self, df):
        self.mean = df.SalePrice.mean()
        self.year_market_strength = df.groupby('SaleYear').mean().SalePrice / self.mean
        self.year_market_strength.loc[2012] = 2*self.year_market_strength.loc[2011] \
                                            - 1*self.year_market_strength.loc[2010]

        df['MarketScaledPrice'] = df['SalePrice'].values.reshape(-1,1) /   \
                                self.year_market_strength.loc[df['SaleYear']].values.reshape(-1,1)

        #self.model_means = df.groupby('ModelId').mean().SalePrice
        self.MIDinterpolators = {}
        for MID in df.ModelID.unique():
            mdf = df[df.ModelID == MID]
            X = mdf.YearsFrom2012
            y = mdf.MarketScaledPrice
            interp = Ridge(alpha = 1, fit_intercept=True, normalize=True)
            X=X.values.reshape(-1,1)
            y=y.values.reshape(-1,1)
            #XS = np.concatenate([X, X**2, X**3, X**4],axis=1)
            XS = np.exp(-X)
            interp.fit(XS,y)
            self.MIDinterpolators[MID] = interp

        self.group_means = df.groupby('ProductGroup').mean().MarketScaledPrice

    def __call__(self, df):
        #ipdb.set_trace()
        df['InterpIntercept'] = 0
        for MID in self.MIDinterpolators:
            if MID in df.ModelID.values:
                interp = self.MIDinterpolators[MID]
                inds = df[df.ModelID==MID].index
                X = df.loc[inds,'YearsFrom2012'].values.reshape(-1,1)
                #Xs = np.concatenate([X, X**2, X**3, X**4],axis=1)
                XS = np.exp(-X)
                df.loc[inds,'InterpIntercept'] += interp.predict(Xs).flatten()

        zdf = df[df['InterpIntercept']==0]
        for cat in self.group_means.index.values:
            inds = zdf[zdf.ProductGroup==cat].index
            df.loc[inds,'InterpIntercept'] += self.group_means.loc[cat]
        #ipdb.set_trace()
        for years in self.year_market_strength.index.values:
            inds = df[df.YearsFrom2012==years].index
            df.loc[inds,'InterpIntercept'] /= self.year_market_strength.loc[years]
        return df

class Preprocessor:
    def __init__(self, df):

        # remember nan status of training cols:
        # self.has_nulls = df.columns[df.isnull().any()].values
        # self.all_nulls = df.columns[df.isnull().all()].values

        self.target_cols = np.array(['SalePrice', 'MarketScaledPrice'])
        self.index_cols  = np.array(['SalesID','ModelID','MachineID','ProductGroup',
                'SaleDate','SaleYear', 'YearMade','YearsFrom2012'])
        self.intercept_col = np.array(['InterpIntercept'])
        self.feature_cols = diff( df.columns.values,
                                    reduce(union,(self.target_cols, self.index_cols, self.intercept_col)))
        #ipdb.set_trace()
        # organize features into categorical, numeric and other:
        object_cols = inter ( df.columns[df.dtypes == np.object].values, self.feature_cols )
        self.object_cols = np.array(object_cols)
        self.numeric_cols = diff(self.feature_cols, self.object_cols)


        # set up regularization process:
        # (dictionary of col_name: Regularizer objects)
        self.regularizers = {col: Regularizer(df[col]) for col in self.numeric_cols}

        self.encoders = {}
        for col in self.object_cols:
            # if col == "ModelID":
            #     self.encoders[col] = OneHotEncoder(df,col,500)
            # else:
            self.encoders[col] = OneHotEncoder(df,col,2)



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
