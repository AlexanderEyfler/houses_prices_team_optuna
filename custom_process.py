from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        replace_map = {
            'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0,
            'Grvl': 2, 'Pave': 1,
            'Y': 2, 'P': 1, 'N': 0,
            'Av': 3, 'Mn': 2, 'No': 1,
            'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1,
            'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1,
            'Fin': 3, 'RFn': 2
        }
        values_to_check = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA',
                           'Grvl', 'Pave', 
                           'Y', 'P', 'N',
                           'Av', 'Mn', 'No',
                           'GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf',
                           'GdPrv', 'MnPrv', 'GdWo', 'MnWw',
                           'Fin', 'RFn']
        
        columns_with_values = [
            col for col in X.columns
            if X[col].isin(values_to_check).any()
        ]
        X[columns_with_values] = X[columns_with_values].replace(replace_map)

        return X

class MissingValueReplacer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X.replace('Missing', 0, inplace=True)
        return X