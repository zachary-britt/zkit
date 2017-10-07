from sklearn.linear_model import LinearRegression

class ModelPoser:
    def __init__(self, X, y):
        self.LR = LinearRegression(fit_intercept=True,
            normalize=True, njobs=-1)
        self.LR.fit(X,y)

    def predict(self, X):
        return self.LR.predict(X)

    def score(X,y):
        return self.LR.score(X,y)

    def get_coeff():
        coeff = [self.LR.intercept_]
        coeff.extend(self.LR.coef_)
        return coeff
