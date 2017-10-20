import pandas as pd
import numpy as np
import sys
import pickle
import ipdb
import argparse

from data_loader import data_loader, feature_saver, feature_loader
from preprocessor import Preprocessor
from feature_builder import FeatureBuilder
from model_poser import ModelPoser
from visualizers import explore_viz_np, explore_viz_pd, explain_viz

def build_arg_parser():
    # use command line to prevent unnecessary computations

    parser = argparse.ArgumentParser(description='Process command line options.')

    parser.add_argument('--have_df', dest='have_df', action='store_const',
                        const=True, default=False,
                        help='Run the cached version of the dataframe (default: rebuild the dataframe)')

    parser.add_argument('--have_features', dest='have_features', action='store_const',
                        const=True, default=False,
                        help='Run the cached features (default: rebuild features)')

    parser.add_argument('--run_model', dest='run_model', action='store_const',
                        const=True, default=False,
                        help='Run the model builder on the data (default: exit without building)')

    parser.add_argument('--run_test', dest='run_test', action='store_const',
                        const=True, default=False,
                        help='Run the model on the test set (default: exit without test)')

    parser.add_argument('--run_plot', dest='run_plot', action='store_const',
                        const=True, default=False,
                        help='Run the visualizers (default: dont)')

    return parser

if __name__ == "__main__":

    parser=build_arg_parser()
    args = parser.parse_args()

    if args.have_features == False:

        #Load dataframes. Proportion is the porportion of your data to load.
        df_train, df_test = data_loader(args.have_df, proportion=0.05)



        if args.run_plot:
            # Early Plotting

            # plot_MID_interps(df_train)
            pass


        # build cleaning process
        preprocessor = Preprocessor(df_train)

        #execute cleaning process
        df_train = preprocessor(df_train)
        df_test = preprocessor(df_test)


        #build and store feature engineering process
        f_builder = FeatureBuilder(df_train)

        #execute feature engineering
        y_train = df_train.pop('churn').values
        y_test = df_test.pop('churn').values
        X_train, feature_names = f_builder.engineer_features(df_train)
        X_test, feature_names = f_builder.engineer_features(df_test)

        feat_list = [X_train, y_train, X_test, y_test, feature_names]
        feature_saver(feat_list)

    else: #have features is true
        X_train, y_train, X_test, y_test, feature_names = feature_loader

    if args.run_model: # (command line arg)
        # build models, perform parameter optimization and save best model
        model_poser = ModelPoser(X_train, y_train)
        model = model_poser.get_best()

        # print(model.get_score())
        # print(model.get_params())
        # print(model.get_coeff())
        # print(model.grid_search.cv_results_)

        # model = model_maker(X_train, y_train)


    if run_test: # (command line arg)
        #execute model on test set
        y_pred = model.predict(X_test)

        #save results to cache
        with open('cache/y.npy','wb') as f: np.save(f,y_pred)
        with open('cache/x.npy','wb') as f: np.save(f,X_test)
        with open('cache/model.pkl','wb') as f: pickle.dump(model, f)
        with open('cache/feature_names.pkl','wb') as f: pickle.dump(feature_names, f)
