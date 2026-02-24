### Load necessary modules -------------------------------


# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set()
import statsmodels.api as sm


# Data management libraries
import numpy as np 
import pandas as pd
import scipy.stats as stats

# Machine learning libraries
from sklearn.metrics import root_mean_squared_error 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# Scikit-learn regression models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.inspection import permutation_importance

# Others
import warnings


# Let us start by loading the dataset and taking a first look at it.

# %%
df = pd.read_csv("../3_3_Beyond_Linear_Models/diamonds2.csv")


# %%
num_inputs = ["carat", "depth", "table", "x", "y", "z", "volume"]
cat_inputs = ["cut", "color", "clarity"]

inputs = num_inputs + cat_inputs

output = "log_price"
 
# ### Split and Standard Names for the Datasets

X = df[inputs]
Y = df[output]

XTR, XTS, YTR, YTS = train_test_split(
    X, Y,
    test_size = 0.2,  
    random_state = 1) 

# %%
YTS.shape, XTS.shape

# %%
dfTR = pd.DataFrame(XTR, columns=inputs)
dfTR[output] = YTR


# %%
dfTS = pd.DataFrame(XTS, columns=inputs)
dfTS[output] = YTS


# We end up this initial step by checking the first few rows of `dfTR`.

# %%
dfTR.head()


# ### Dealing with outliers.

# %%
def remove_outliers(X, method="iqr", threshold=1.5):
    if isinstance(X, pd.DataFrame):
        X = X.copy()  # Avoid modifying the original DataFrame
        columns = X.columns  # Save column names
        idx = X.index  # Save row index
        X_values = X.values  # Work on raw numpy array

    else:
        return X  # If not a DataFrame, return as is

    if method == "iqr":
        Q1 = np.nanpercentile(X_values, 25, axis=0)
        Q3 = np.nanpercentile(X_values, 75, axis=0)
        IQR = Q3 - Q1
        mask = (X_values >= Q1 - threshold * IQR) & (X_values <= Q3 + threshold * IQR)
    elif method == "std":
        mean = np.nanmean(X_values, axis=0)
        std = np.nanstd(X_values, axis=0)
        mask = (X_values >= mean - threshold * std) & (X_values <= mean + threshold * std)
    else:
        return X  # If the method is not valid, return without changes

    X_values = np.where(mask, X_values, np.nan)  # Convert outliers to NaN
    return pd.DataFrame(X_values, columns=columns, index=idx)  # Convert back to DataFrame




def create_outlier_transformer(threshold=1.5, method="iqr", use_outlier_removal = True):
    if use_outlier_removal:
         return FunctionTransformer(lambda X: remove_outliers(X, method=method, threshold=threshold))
    else:
        return FunctionTransformer(lambda X: X)

# Create the transformer with an initial threshold
outlier_remover = create_outlier_transformer(threshold=1.5, use_outlier_removal = True)
outlier_remover.set_output(transform='pandas')



# ### Column transformer
# 

# %%
outlier_removal = ColumnTransformer(
    transformers=[("outlier_remover", outlier_remover, num_inputs)],
    remainder="passthrough", 
    force_int_remainder_cols=False)
outlier_removal.set_output(transform='pandas')



# ### Pipeline and the issue of Scikit naming conventions

# %%
preproc_pipeline = Pipeline([
    ("remove_outliers", outlier_removal)
])

preproc_pipeline.set_output(transform='pandas')


# ### Restoring the original names of the variables

class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, separator="__"):
        self.separator = separator
        self._output_format = 'pandas'  # Default output format


    def fit(self, X, y=None):
        return self  # Nothing to learn, just transforming

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Remove prefix up to the separator "__"
            new_columns = [col.split(self.separator, 1)[-1] if self.separator in col else col for col in X.columns]
            X = X.copy()
            X.columns = new_columns
        return X

    def set_output(self, transform='pandas'):
        """Allows the user to choose the output format for the transformer."""
        if transform not in ['array', 'pandas']:
            raise ValueError("Output format must be 'array' or 'pandas'")
        self._output_format = transform



# %%
new_step = ("outliers_name_cleanup", ColumnNameCleaner())
preproc_pipeline = Pipeline(preproc_pipeline.steps + [new_step])
preproc_pipeline



# ## Dealing with missing values.  Imputation methods.

# ### Implementing the SimpleImputer

from sklearn.impute import SimpleImputer

num_imputer = (SimpleImputer(strategy="median"))
num_imputer.set_output(transform='pandas')

cat_imputer = SimpleImputer(strategy="most_frequent")
cat_imputer.set_output(transform='pandas')


# %%
imputer = ColumnTransformer([
    ("num_imputer", num_imputer, num_inputs),
    ("cat_imputer", cat_imputer, cat_inputs)
])

imputer.set_output(transform='pandas'); # the semicolon suppresses the output diagram, that we will see below

# %%
new_steps = [("imputer", imputer), ("imputer_name_cleanup", ColumnNameCleaner())]
preproc_pipeline = Pipeline(preproc_pipeline.steps + new_steps)
preproc_pipeline


# ## One Hot Encoding

# %%
ohe = OneHotEncoder(sparse_output=False, drop="first")
ohe.set_output(transform='pandas')  


# Defining the associated column transformer is as usual.

# %%
encoder = ColumnTransformer([("ohe", ohe, cat_inputs)],
                            remainder="passthrough")
encoder.set_output(transform='pandas')


# And we can now update the pipeline with this one hot encoding step.
# %%
new_step = ("encoder", encoder)
preproc_pipeline = Pipeline(preproc_pipeline.steps + [new_step])

# %%
new_step = ("encoder_name_cleanup", ColumnNameCleaner())
preproc_pipeline = Pipeline(preproc_pipeline.steps + [new_step])

# ## Scaling the numerical inputs


# %%
def create_scaler(use_scaler = True):
    if use_scaler:
         return StandardScaler()
    else:
        return FunctionTransformer(lambda X: X)

# Create the transformer with an initial threshold
scaler = create_scaler(use_scaler = True)
scaler.set_output(transform='pandas')

# %%
scaling = ColumnTransformer([
    ("scaler", scaler, num_inputs)
],
    remainder="passthrough")
scaling.set_output(transform='pandas')

new_steps = [("scaler", scaling),
             ("scaler_name_cleanup", ColumnNameCleaner())]

preproc_pipeline = Pipeline(preproc_pipeline.steps + new_steps)


# ## Dealing with collinearity in numerical inputs


# %%
XTR_transf = preproc_pipeline.fit_transform(XTR)



# ### A correlation matrix based transformer
# %%
class DropHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, varOrder = None):
        self.threshold = threshold
        self.varOrder = varOrder
        self.original_columns = None
        self._output_format = 'array'  # Default output format

    def fit(self, X, y=None):
        self.original_columns = X.columns
        if self.varOrder is not None:            
            X = X[self.varOrder]
        # Compute the correlation matrix
        # print("column order used here: ", X.columns)
        self.corr_matrix = X.corr().abs()
        return self

    def transform(self, X):
        # Identify columns to drop
        columns_to_drop = set()
        for i in range(len(self.corr_matrix.columns) + 1):
            for j in range(i, len(self.corr_matrix.columns)):
                # print(f"i = {i}, j = {j}, corr = {self.corr_matrix.iloc[i, j]}")
                if j > i and self.corr_matrix.iloc[i, j] > self.threshold:
                    colname = self.corr_matrix.columns[i]
                    # print(f"Column {colname} is highly correlated with {self.corr_matrix.columns[j]} and marked for removal")
                    columns_to_drop.add(colname)
        
        # Drop highly correlated columns
        filtered_data = X[self.original_columns].drop(columns=columns_to_drop)
        
        # If the output format is 'pandas', convert to pandas DataFrame
        if self._output_format == 'pandas':
            return pd.DataFrame(filtered_data, columns=filtered_data.columns)
        
        # Default is to return numpy array
        return filtered_data

    def set_output(self, transform='array'):
        """Allows the user to choose the output format for the transformer."""
        if transform not in ['array', 'pandas']:
            raise ValueError("Output format must be 'array' or 'pandas'")
        self._output_format = transform




# %%
def create_collinearity_remover(use_collinearity_removal = True, threshold=0.9, varOrder = None):
    if use_collinearity_removal:
         return DropHighlyCorrelatedFeatures(threshold=threshold, varOrder=varOrder)
    else:
        return FunctionTransformer(lambda X: X)


# %%
use_collinearity_removal  = True
threshold = 0.92

collinearity_remover = create_collinearity_remover(threshold=threshold,     
                                                   varOrder = ['depth', 'table', 'volume', 'z', 'y', 'x', 'carat'], 
                                                #    varOrder = ['x', 'y', 'z', 'carat'], 
                                                   use_collinearity_removal = use_collinearity_removal)
collinearity_remover.set_output(transform='pandas')


remove_collinearity = ColumnTransformer([
    ("collinearRemover", collinearity_remover, num_inputs)
],
    remainder="passthrough")
remove_collinearity.set_output(transform='pandas')


new_steps = [("remove_collinearity", remove_collinearity), 
             ("name_cleanup_collinear", ColumnNameCleaner())
             ]
preproc_pipeline = Pipeline(preproc_pipeline.steps + new_steps)

# %%
XTR_transf = preproc_pipeline.fit_transform(XTR)

num_inputs_left = [col for col in XTR_transf.columns if col in num_inputs]
num_inputs_left



# ## Transformations to increase normality

# %%
use_power_transformer = True

def create_power_transformer(use_power_transformer = use_power_transformer, method='yeo-johnson', standardize=True):
    if use_power_transformer:
         return PowerTransformer(method=method, standardize=standardize) 
    else:
        return FunctionTransformer(lambda X: X)


power_transformer = create_power_transformer(use_power_transformer = True, method='yeo-johnson', standardize=True)
power_transformer.set_output(transform='pandas')

# Now the rest of the code is easy:

# %%
power_transformed_vars = num_inputs_left

power_transform = ColumnTransformer([
    ("power_transformer", power_transformer, num_inputs_left)
],
    remainder="passthrough")
power_transform.set_output(transform='pandas')


new_steps = [("power_transform", power_transform), 
             ("name_cleanup_power_transform", ColumnNameCleaner())
             ]
preproc_pipeline = Pipeline(preproc_pipeline.steps + new_steps)






# ## Adding polynomial features


# %%
# Leave the list of variables empty to skip this step
poly_vars = ['x']
poly_degrees = [2]

column_transformations = []
for var, degree in zip(poly_vars, poly_degrees):
    var_polfeat = PolynomialFeatures(degree=degree, include_bias=False)
    var_polfeat.set_output(transform='pandas')
    # Here we need a list for the last argument, otherwise it will be considered as a string
    # and pandas in its infinite wisdom will return a Series instead of a DataFrame 
    new_col_transformer = (f"poly_{var}", var_polfeat, [var]) 
    column_transformations.append(new_col_transformer)
    
# %%
polyterms_transform = ColumnTransformer(column_transformations, remainder="passthrough")
polyterms_transform.set_output(transform='pandas')
polyterms_transform


# %%

new_steps = [("add_polyterms", polyterms_transform), 
             ("name_cleanup_polyterms", ColumnNameCleaner())]
preproc_pipeline = Pipeline(preproc_pipeline.steps + new_steps)




