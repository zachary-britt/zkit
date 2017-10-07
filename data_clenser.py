import pandas as pd
import numpy as np

class DataClenser:
    def __init__(self, df):
        self.has_nulls = df.columns[df.isnull().any()].tolist()
        self.all_nulls = df.columns[df.isnull().all()].tolist()

    def clean(df):
        df = df.drop(self.has_nulls, axis=1)
        return df
