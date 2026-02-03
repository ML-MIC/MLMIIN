import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split

df = pd.read_csv('../2_2_Classification_Logistic_Regression/Default.csv')

df.drop(columns=["student", "income"], inplace=True)
df["default"] = df["default"].astype("category")

output = 'default'
cat_inputs = []
df[cat_inputs + [output]] = df[cat_inputs + [output]].astype("category")

inputs = df.columns.drop(output)
num_inputs = inputs.difference(cat_inputs).tolist()
inputs

df1 = df.sample(n = 1200, random_state=1)

X = df1.drop(columns=output)
Y = (df1[output] == "Yes") * 1.0

XTR, XTS, YTR, YTS = train_test_split(X, Y,
                                      test_size=0.2,  # percentage preserved as test data
                                      random_state=1, # seed for replication
                                      stratify = Y)   # Preserves distribution of y