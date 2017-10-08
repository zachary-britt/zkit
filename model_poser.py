import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
import pdb

class ModelPoser:
    def __init__(self, X, y):



        '''
        [mean: 0.64011, std: 0.07320, params: {'alpha': 0.25},
        mean: 0.64250, std: 0.07455, params: {'alpha': 0.5},
        mean: 0.64428, std: 0.07646, params: {'alpha': 1.0},
        mean: 0.64366, std: 0.07834, params: {'alph]
        '''

        param_space = {'alpha': np.logspace(2,4,3, base=10)}
        lasso = Lasso(tol=0.0001)
        grid_search = GridSearchCV(lasso, param_space, cv=4, n_jobs=-1)

        grid_search.fit(X,y)
        # print(grid_search.grid_scores_)
        self.grid_search=grid_search
        self.params = grid_search.best_params_
        self.score = grid_search.best_score_
        self.best = grid_search.best_estimator_
        pdb.set_trace()
        return

    def predict(self, X):
        return self.best.predict(X)

    def get_score(self):
        return self.grid_search.best_score_

    def get_params(self):
        return self.grid_search.best_params_

    def get_coeff(self):
        coeff = [self.best.intercept_]
        coeff.extend(self.best.coef_)
        return coeff
