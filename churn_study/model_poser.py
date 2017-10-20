import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC



from sklearn.model_selection import cross_val_score, GridSearchCV
import ipdb

class ModelPoser:
    def __init__(self, X, y):

        model_list = [RF, LR, GBC, SVC(probability=True)]


        #param_space = {'alpha': np.logspace(-1,2,4, base=2)}

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
