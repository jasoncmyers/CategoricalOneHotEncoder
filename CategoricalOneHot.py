from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    '''
    A stateful transformer for converting string categorical columns into one-hot encoded columns.
    Currently uses pd.get_dummies() and drops extra columns from the test set at the end, so it isn't the most
    efficient thing in the world.  But ideally I'll be able to use the newer version of sklearn where OneHotEncoder handles
    this before too long, so it isn't worth optimizing too much.  Based on the sklearn OneHotEncoder source.
    
    Parameters
    ----------
    sparse : bool, default=False
        Should the function return a NumPy array (False) or a sparse matrix (True)
        
    dtype: NumPy dtype, default=uint8
        The dtype for the encoded values (1 or 0)
        
    handle_unknown : {'error', 'ignore'}, default='error'
        How should the transformer react if it encounters unknown categories during the transfor
        (i.e., categories not present in the fit() dataset).  Default raises an error.  Ignore will discard 
        any new categories (any value not 'error' will be treated as 'ignore').
    
    '''
    
    def __init__(self, *, sparse=False, dtype=np.uint8, handle_unknown='error'):
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.fitted_columns = None
    
    
    def fit(self, X, y=None):
        
        X_trans = pd.get_dummies(X)
        self.fitted_columns = set(X_trans.columns)
                
        return self
    
    
    def fit_transform(self, X, y=None):
        
        X_trans = pd.get_dummies(X, sparse=self.sparse)
        self.fitted_columns = set(X_trans.columns)
        
        return X_trans
    
    
    def transform(self, X, y=None):
        
        X_trans = pd.get_dummies(X, sparse=self.sparse)
        
        # check for any columns not in the original fit()
        extra_cols = set(X_trans.columns) - self.fitted_columns
        
        # raise errors for extra columns, if that's what it's supposed to do
        if self.handle_unknown == 'error':
            if len(extra_cols) != 0:
                raise ValueError(
                    'Features found that were not present in the fit() data: {}'.format(extra_cols)
                )
        else:
            # drop any columns not in the fit() set
            X_trans = X_trans.drop(extra_cols, axis=1)
            
            # add in any missing columns and make column order match
            missing_cols = self.fitted_columns - set(X_trans.columns)
            for col in missing_cols:
                X_trans[col] = 0
                
        return X_trans