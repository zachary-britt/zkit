import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
