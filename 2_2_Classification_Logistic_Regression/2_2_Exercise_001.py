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
df = pd.read_csv('Simdata0.csv', sep=";")
# df.head(4)
# df.info()

# Remove Unused Columns from the Dataset
df.drop(columns="id", inplace=True) 
# df.head()

# Think about the proper type for each variable
# df.nunique()

# Set the type of all categorical variables and use fixed names
output = 'Y'
cat_inputs = ['X4']
df[cat_inputs + [output]] = df[cat_inputs + [output]].astype("category")

# Similarly for numerical inputs and create an index of all input variables.

inputs = df.columns.drop(output)
num_inputs = inputs.difference(cat_inputs).tolist()
# print(inputs)
# df.info()


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

# The `stratify = y` option guarantees that the proportion of both output classes is (approx.) the same in the trainng and test datasets. 
# <br><br><br>

# ### Exercise 002.
# Check this statement by making a frequency table for the output classes in training and test. 


# # 2 Preproc; Step 5: Plot the numeric variables in training data and check out for outliers  {.unnumbered .unlisted}

# **Outliers** (also called *atypical values*) can be generally defined as *samples that are exceptionally far from the mainstream of the data*. Formally, a value $x$ of a numeric variable $X$ is considered an outlier if it is bigger than the *upper outlier limit*
# $$q_{0.75}(X) + 1.5\operatorname{IQR}(X)$$ 
# or if it is smaller than the *lower outlier limit* 
# $$q_{0.25}(X) - 1.5\operatorname{IQR}(X),$$ 
# where $q_{0.25}(X), q_{0.45}(X)$  are respectively the first and third **quartiles** of $X$ and $\operatorname{IQR}(X)$ is the **interquartilic range** of $X$. 
# ![](./fig/outliers_Imagen%20de%20MachineLearning_Ch2_Classification_1_Intro_preprocess_v3_%20p19.png){width=40% fig-align="center" fig-alt="Outliers"}
# A boxplot graph, like the one on the left, of the variable is often the easiest wy to spot the pressence of outliers.



# Outliers processing, Boxplots for the Numerical Inputs

# XTR_numeric_boxplots = XTR[num_inputs].plot.box(subplots=True, 
#                                             layout=(1, len(num_inputs)), 
#                                             sharex=False, sharey=False, figsize=(6, 3))


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

# Check result
# explore_outliers(XTR, num_inputs)


# EDA

# Describe for numerical inputs

# XTR[num_inputs].describe()

# Describe for factor inputs

# XTR[cat_inputs].describe().transpose()

# Pairplots

# sns.set_style("white")
# plt_df = XTR[num_inputs].copy()
# plt_df["YTR"] = YTR
# sns.pairplot(plt_df, hue="YTR", corner=True, height=2)

# Plots to Study Numerical Input vs Factor Input Relations

# for numVar in num_inputs: 
#     print("Analyzing the relation between factor inputs and", numVar)
#     fig, axes = plt.subplots(1, len(cat_inputs))  # create figure and axes
#     for col, ax in zip(cat_inputs, axes):  # boxplot for each factor inpput
#         sns.boxplot(data=XTR, x = col, y = numVar, ax=ax) 
#     # set subplot margins
#     plt.subplots_adjust(left=0.9, bottom=0.4, right=2, top=1, wspace=1, hspace=1)
#     plt.figure(figsize=(1, 1))
#     plt.show()

# Correlation Matrix

XTR[num_inputs].corr()

# Visualization of the Correlation Matrix
# plt.figure()
# plt.matshow(XTR[num_inputs].corr(), cmap='viridis')
# plt.xticks(range(len(num_inputs)), num_inputs, fontsize=14, rotation=45)
# plt.yticks(range(len(num_inputs)), num_inputs, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()

# Use of the Correlation Matrix for Feature Selection
XTR.drop(columns="X3", inplace=True)
num_inputs.remove("X3") # Keep the list of inputs updated 

# Dropping the Initial Categorical Inputs

XTR.drop(columns=cat_inputs, inplace=True)


# Check Imbalance

# YTR.value_counts(normalize=True)

print("Preprocessing completed. Train and test set created.")

