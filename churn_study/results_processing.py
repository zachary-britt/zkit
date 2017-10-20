'''
Make insightful statements on model performance
'''


import pandas as pd
import numpy as np
import sys
import pickle
import ipdb
import argparse

from model_poser import ModelPoser
from visualizers import explore_viz_np, explore_viz_pd, explain_viz


def confusion_classifier(y_pred, y_test):
    TP = np.sum((y_pred==True) & (y_test==True))
    FP = np.sum((y_pred==True) & (y_test==False))
    FN = np.sum((y_pred==False) & (y_test==True))
    TN = np.sum((y_pred==False) & (y_test==False))
    return TP, FP, FN, TN

def confusion_matrix_builder(TP, FP, FN, TN):
    return np.array([[TP,FP],[FN,TN]])

def total_profit(confusion_matrix, profit_matrix):
    return np.tensordot(confusion_matrix, profit_matrix)



if __name__ == "__main__":

    y_pred = np.load('cache/y.npy')
    X_test = np.load('cache/x.npy')

    '''FIX ME'''
    y_test = np.ones(y_pred.shape)

    with open('cache/model.pkl','rb') as f: model = pickle.load(f)


    TP, FP, FN, TN = confusion_classifier(y_pred, y_test)
    confusion_matrix = confusion_matrix_builder(TP, FP, FN, TN)

    '''FIX ME'''
    profit_matrix = np.array([[1,-5],[-2, 0]])







#
