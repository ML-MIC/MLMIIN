# %load  "../2_2_Classification_Logistic_Regression/2_2_Exercise_001.py"
# Import standard Python Data Science libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from matplotlib.cbook import boxplot_stats

# Load the data
df = pd.read_csv('SimData0.csv', sep=";")

# Remove Unused Columns from the Dataset
df.drop(columns="id", inplace=True) 

# Set the type of all categorical variables and use fixed names
output = 'Y'
cat_inputs = ['X4']
df[cat_inputs + [output]] = df[cat_inputs + [output]].astype("category")

# Similarly for numerical inputs and create an index of all input variables.

inputs = df.columns.drop(output)
num_inputs = inputs.difference(cat_inputs).tolist()

# One Hot Encoding of the Categorical Inputs

ohe = OneHotEncoder(sparse_output=False)
ohe_result = ohe.fit_transform(df[cat_inputs])
ohe_inputs = [ohe.categories_[k].tolist() for k in range(len(cat_inputs))]
ohe_inputs = [cat_inputs[k] + "_" + str(a) for k in range(len(cat_inputs)) for a in ohe_inputs[k]]
ohe_df = pd.DataFrame(ohe_result, columns=ohe_inputs)
df[ohe_inputs] = ohe_df


# Check out the missing values. We can either drop them or do imputation.
df.isnull().sum(axis=0)
df.dropna(inplace=True)
df.isnull().sum(axis=0)


# Split the dataset into Training and Test Sets
#  Using Standard Names for the Variables

X = df.drop(columns=output)
Y = df[output]

XTR, XTS, YTR, YTS = train_test_split(X, Y,
									  test_size=0.2,  # percentage preserved as test data
									  random_state=1, # seed for replication
									  stratify = Y)   # Preserves distribution of y


# Locate and drop outliers if needed.

def explore_outliers(df, num_vars, show_boxplot=False):
	if show_boxplot:
		fig, axes = plt.subplots(nrows=len(num_vars), ncols=1, figsize=(7, 3), sharey=False, sharex=False)
		fig.tight_layout()
	outliers_df = dict()
	for k in range(len(num_vars)):
		var = num_vars[k]
		if show_boxplot:
			sns.boxplot(df, x=var , ax=axes[k])
		outliers_df[var] = boxplot_stats(df[var])[0]["fliers"]
		out_pos = np.where(df[var].isin(outliers_df[var]))[0].tolist() 
		out_idx = [df[var].index.tolist()[ k ] for k in out_pos]
		outliers_df[var] = {"values": outliers_df[var], 
							"positions": out_pos, 
							"indices": out_idx}
	return outliers_df

# Use this function with the numeric inputs in XTR

out_XTR = explore_outliers(XTR, num_inputs)
# out_XTR

# Then use the result to drop all the outliers. 

out_XTR_indices = set([k for var in num_inputs for k in out_XTR[var]["indices"] ])
XTR.drop(out_XTR_indices, axis=0, inplace=True)

# Also make sure to remove the corresponding output values in YTR

YTR.drop(out_XTR_indices, axis=0, inplace=True) # Always keep in mind that you need to keep `YTR` updated.


# Dropping the Initial Categorical Inputs

XTR.drop(columns=cat_inputs, inplace=True)


print("Preprocessing completed. Train and test set created.")
