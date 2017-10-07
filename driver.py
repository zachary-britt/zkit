import pandas as pd
import numpy as np
import sys
import pickle

from data_loader import data_loader
from data_cleanser import DataClenser
from feature_builder import FeatureBuilder
from model_poser import ModelPoser
from visualizers import explore_viz, explain_viz


if __name__ == "__main__":

    # use command line to prevent unnecessary computations
    if len(sys.argv) > 1:
        command_args = set(sys.argv[1:])
        run_test = 'predict' in command_args
        run_cached_predictions = 'have_preds' in command_args
        run_plotting = 'plot' in command_args
    else:
        run_test = False
        run_cached_predictions = False
        run_plotting = False

    if not run_cached_predictions: # (command line arg)

        # load dataframes
        # optionally load random subset of data to reduce runtime while testing
        df_train = data_loader('data/Train.csv', proportion=0.01)
        if run_test:
            df_test = data_loader('data/Test.csv', proportion=1)

        #build and store cleaning process
        d_cleaner = DataClenser(df_train)

        #execute cleaning process
        df_train = d_cleaner.clean(df_train)
        df_test = d_cleaner.clean(df_test)

        #build and store feature engineering process
        f_builder = FeatureBuilder(df_train)

        #execute feature engineering
        X_train, y_train, names = f_builder.engineer_features(df_train)
        X_test, y_test, names = f_builder.engineer_features(df_test)

        if run_plotting: # (command line arg)
            #for simple exploratory data visualization
            explore_viz(X_train, y_train, names)


        #build models, perform parameter optimization and save best model
        model = ModelPoser(X_train, y_train)

        if run_test: # (command line arg)
            #execute model on test set
            y_pred = model.predict(X_test)

            #print 'score' results.
            print(model.score(X_test, y_test))
            model_coeff = model.get_coeff()
            print(model_coeff)

            #save results to cache
            ys = np.concatenate([y_test.reshape(-1,1), y_pred.reshape(-1,1)],
                axis = 1)
            np.savetxt('data/y_cache.txt', ys)
            np.savetxt('data/x_cache.txt', X_test)
            np.savetxt('data/model_coeff.txt', model_coeff)
            np.savetxt('data/feature_names.txt', names)
            # with open('data/model_cache.pkl', 'wb') as pkl_file1:
            #     pickle.dump(model, pkl_file1)
            # with open('data/names.pkl', 'wb') as pkl_file2:
            #     pickle.dump(names, pkl_file2)

    else: # run_cached_predictions == True  (command line arg)
        #load predictions from cache
        ys = np.loadtxt('data/y_cache.txt')
        y_test, y_pred = ys[:,0], ys[:,1]
        X_test = np.loadtxt('data/x_cache.txt')
        model_coeff=np.loadtxt('data/model_coeff.txt')
        names=np.loadtxt('data/feature_names.txt')
        # pkl_file1 = open('data/model_cache.pkl', 'rb')
        # model = pickle.load(pkl_file1)
        # pkl_file1.close()
        # pkl_file2 = open('data/names.pkl', 'rb')
        # names = pickle.load(pkl_file2)
        # pkl_file2.close()

        #produce results visualizations
        explain_viz(X_test, y_test, y_pred, model, names)











    #buffer
