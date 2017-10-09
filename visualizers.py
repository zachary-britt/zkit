import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import pdb

def price_of_cat_vs_date(df, X):
    pass

def plot_MID_interps(df):

    fig = plt.figure(figsize=(10,10))

    rs=6
    cs=6

    for r in range(rs):
        for c in range(cs):
            MID = df.ModelID.value_counts().index.values[r*rs+c+1000]
            mdf = df[df.ModelID == MID]
            X = mdf.YearsFrom2012
            y=mdf.SalePrice
            ax = fig.add_subplot(rs,cs,(r)*rs+(c+1))
            plot_time_interpolator(X, y, ax)
    plt.tight_layout()
    plt.show()

def plot_time_interpolator(X,y,ax):
    interp = Ridge(alpha = 1, fit_intercept=True, normalize=True)
    X=X.values.reshape(-1,1)
    y=y.values.reshape(-1,1)
    XS = np.concatenate([X, X**2, X**3, X**4],axis=1)
    interp.fit(XS,y)
    x = np.linspace(min(X)[0], 0, 100).reshape(-1,1)
    xs = np.concatenate([x, x**2, x**3, x**4],axis=1)
    ys = interp.predict(xs)

    ax.scatter(X,y,alpha=0.1)
    ax.plot(x,ys,c='r')



def explore_viz_pd(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    year_prices = df.groupby(['SaleYear']).mean()
    ax.scatter( df['SaleYear'], df['SalePrice'] )
    plt.show()


def explore_viz_np(X, y, names):

    pass

def explain_viz(*args):
    pass
