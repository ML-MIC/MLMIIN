from sklearn.base import BaseEstimator, TransformerMixin
import patsy as ps
import pandas as pd

class FormulaTransformer(BaseEstimator, TransformerMixin):
    

    def __init__(self, formula):

        self.formula = formula
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_idx = X.index
        X_formula = ps.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        return pd.DataFrame(X_formula, columns=columns, index = X_idx)
        
       