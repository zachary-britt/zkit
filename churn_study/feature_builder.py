
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

        names = df.columns.values
        X = df.values
        return X, names
