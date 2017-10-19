import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
import ipdb

class ModelPoser:
    def __init__(self, X, y):

        '''
        'mean_test_score':
        array([
        0.62718964,     lasso alpha = 1
        0.62170231,     4
        0.59540675,     16
        0.56490968,     64
        0.52374966]),   256


        0.63816927,     net alpha = 0.5, ratio = 0.999
        0.62711329,     0.5, 0.999
        '''
        #param_space = {'alpha': np.logspace(-1,2,4, base=2)}
        param_space = {'alpha': [0.5]}
        # lasso = Lasso(max_iter=10000, tol=0.001)
        # grid_search = GridSearchCV(lasso, param_space, cv=4, n_jobs=-1)

        net = ElasticNet(max_iter=10000, tol=0.0001, l1_ratio = 0.999)
        grid_search = GridSearchCV(net, param_space, cv=5, n_jobs=-1)

        grid_search.fit(X,y)
        print(grid_search.cv_results_)
        self.grid_search=grid_search
        self.params = grid_search.best_params_
        self.score = grid_search.best_score_
        self.best = grid_search.best_estimator_
        #ipdb.set_trace()
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

def model_maker(X, y):
    model=ElasticNet(alpha=0.5,l1_ratio=0.999, max_iter=10000, tol=0.00001)
    model.fit(X,y)
    return model
