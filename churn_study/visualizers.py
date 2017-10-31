import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipdb

# from sklearn.linear_model import Ridge, LinearRegression


def explore_viz_pd(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    year_prices = df.groupby(['SaleYear']).mean()
    ax.scatter( df['SaleYear'], df['SalePrice'] )
    plt.show()

if __name__ == "__main__":
    pass
    #put test code here






# def plot_MID_interps(df):
#
#     fig = plt.figure(figsize=(12,8))
#
#     rs=3
#     cs=4
#
#     for r in range(rs):
#         for c in range(cs):
#             MID = df.ModelID.value_counts().index.values[r*cs+c+1010]
#             mdf = df[df.ModelID == MID]
#             X = mdf.YearsFrom2012
#             y = mdf.MarketScaledPrice
#             # y = mdf.SalePrice
#             ys = mdf.InterpIntercept
#             ax = fig.add_subplot(rs,cs,(r)*cs+(c+1))
#             plot_time_interpolator(X, y, ax)
#
#     # fig.suptitle("Price v Time on ModelIDs")
#     # plt.tight_layout()
#     # fig.subplots_adjust(wspace=0.5,hspace=0.2)
#
#     plt.show()

# def plot_time_interpolator(X,y,ax ):
#     interp = Ridge(alpha = 0.2, fit_intercept=True, normalize=True)
#     # interp = LinearRegression(fit_intercept=True, normalize=True)
#     X=X.values.reshape(-1,1)
#     y=y.values.reshape(-1,1)
#     XS = np.concatenate([X, X**2, X**3, X**4],axis=1)
#     # XS = np.exp(-X)
#     interp.fit(XS,y)
#     x = np.linspace(min(X)[0], 0, 100).reshape(-1,1)
#     xs = np.concatenate([x, x**2, x**3, x**4],axis=1)
#     # xs = np.exp(-x)
#     ys = interp.predict(xs)
#
#     ax.scatter(X,y,alpha=0.2)
#     ax.plot(x,ys,c='r')
#     ax.set_ylim(0,max(max(ys),max(y))+0.2*max(y))
#     ax.set_xlim(min(x)-0.3,0)
