
class FeatureBuilder:
    def __init__(self, df):
        '''
        Determine feature building process
        Input: DataFrame
        '''
        pass

    def engineer_features(self, df):
        '''
        Execute feature building process
        Input: DataFrame
        Output: X numpy array,
                y numpy array,
                array of feature names
        '''
        names = ['Intercept']
        names.extend(df.columns.tolist())
        names = np.array(names)
        y = df.pop(df.columns[0]).values
        X = df.values
        return X, y, names
