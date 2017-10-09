import pandas as pd
import numpy as np
import sys
import pickle
import pdb

from data_loader import data_loader
from preprocessor import Preprocessor, make_immediate_drops, add_age_col
from feature_builder import FeatureBuilder
from model_poser import ModelPoser
from visualizers import explore_viz_np, explore_viz_pd, explain_viz


def process_command_line():
    # use command line to prevent unnecessary computations
    if len(sys.argv) > 1:
        command_args = set(sys.argv[1:])
        run_cached_df = 'have_df' in command_args
        run_test = 'run_test' in command_args
        run_cached_predictions = 'have_preds' in command_args
        run_plotting = 'plot' in command_args
        run_model = 'model' in command_args
    else:
        run_cached_df = False
        run_test = False
        run_cached_predictions = False
        run_plotting = False
        run_model = False

    return run_cached_df, run_test, run_cached_predictions, run_plotting, run_model


if __name__ == "__main__":

    run_cached_df, run_test, run_cached_predictions, run_plotting, run_model = process_command_line()

    '''
    Command line args explanation
    run_cached_df:
                    DataFrames are saved to a cache once loaded.
                    Include "have_df" to speed up data loading.
    run_test:
                    Running your test more than once is heresy.
                    Include "run_test" to execute your model on your test set.
    run_cached_predictions:
                    Once you have created a model and ran tests you don't need
                        to do so again (and you shouldn't).
                    Include "have_preds" to implement changes to your
                        explanatory visualization without rerunning the test or
                        rebuilding the model.
    run_plotting:
                    Plots get annoying, but commenting out plt.show() is tacky.
                    Include "run_plotting" to run visualizations
    run_model:
                    Perform Modelling
    '''


    if run_cached_predictions == False:  # (command line arg)

        #Load dataframes. Proportion is the porportion of your data to load.
        df_train, df_test = data_loader(run_cached_df, proportion=1)

        #pdb.set_trace()

        #early cleanup:
        df_train = make_immediate_drops(df_train)
        df_test = make_immediate_drops(df_test)

        #pre-preprocessing
        df_train = add_age_col(df_train)
        df_test = add_age_col(df_test)

        if run_plotting: # (command line arg)
            #for simple exploratory data visualization
            explore_viz_pd(df_train)

        #build and store cleaning process
        preprocessor = Preprocessor(df_train)


        #execute cleaning process
        df_train = preprocessor(df_train)
        df_test = preprocessor(df_test)




        #build and store feature engineering process
        f_builder = FeatureBuilder(df_train)

        #execute feature engineering
        y_train = df_train.pop('SalePrice').values
        X_train, names = f_builder.engineer_features(df_train)
        X_test, names = f_builder.engineer_features(df_test)


        if run_model: # (command line arg)
            #build models, perform parameter optimization and save best model
            model = ModelPoser(X_train, y_train)

            print(model.get_score())
            print(model.get_params())
            print(model.get_coeff())

        if run_test: # (command line arg)
            #execute model on test set
            y_pred = model.predict(X_test)

            #save results to cache
            with open('cache/y.npy','wb') as f: np.save(f,y_pred)
            with open('cache/x.npy','wb') as f: np.save(f,X_test)
            with open('cache/model.pkl','wb') as f: pickle.dump(model, f)
            with open('cache/names.pkl','wb') as f: pickle.dump(names, f)

    else: # run_cached_predictions == True  (command line arg)
        #load from cache

        y_pred = np.load('cache/y.npy')
        X_test = np.load('cache/x.npy')
        with open('cache/model.pkl','rb') as f: model = pickle.load(f)
        with open('cache/names.pkl','rb') as f: names = pickle.load(f)

        print(model.score(X_test, y_test))

        if run_plotting: # (comman line arg)
            #produce results visualizations
            explain_viz(X_test, y_test, y_pred, model, names)



    #buffer
