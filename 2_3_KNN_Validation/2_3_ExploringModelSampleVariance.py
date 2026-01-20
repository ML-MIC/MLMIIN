
'''
2_3_ExploringModelSampleVariance.py
'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


sns.set_theme(rc={'figure.figsize':(12, 4)})

N = 1000000

XP, YP = make_moons(n_samples=N, noise=0.2, random_state=1)

sample_size = 1000

# Select the values of k
k_start = 2
k_stop = 50
k_step = 1

k_values = np.arange(start=k_start, stop=k_stop, step=k_step).astype("int")

for smpl_num in range(0, 5):
    sample = np.random.default_rng(seed=2024).integers(0, N, size=sample_size)

    X = XP[sample, :]
    Y = YP[sample]

    from sklearn.model_selection import train_test_split

    XTR, Xval, YTR, Yval = train_test_split(X, Y,
                                        test_size=0.2,  # percentage preserved as test data
                                        random_state=smpl_num, # seed for replication
                                        stratify = Y)   # Preserves distribution of y


    dfTR = pd.DataFrame(XTR, columns=["X" + str(i + 1) for i in range(X.shape[1])])
    inputs = dfTR.columns
    dfTR["Y"] = YTR
    output = "Y"
    dfval = pd.DataFrame(Xval, columns=inputs)
    dfval["Y"] = Yval

    # Create an empty list to store the accuracies
    accrcies = []
    # Loop through k values, fitting models and getting accuracies
    for k in k_values:
        knn_pipe = Pipeline(steps=[('scaler', StandardScaler()), 
                            ('knn', KNeighborsClassifier(n_neighbors=k))])
        knn_pipe.fit(dfTR[inputs], dfTR[output])

        accrcies.append(knn_pipe.score(dfval[inputs], dfval[output]))
        
    accrcies = np.array(accrcies)
    # Plot accuracies vs k
    ax_acc = sns.scatterplot(x = k_values, y = accrcies)
    sns.lineplot(x = k_values, y = accrcies, ax=ax_acc, label="sample_"+str(smpl_num))
    # Axes labels
    ax_acc.set(xlabel ="k (num. of neighbors)", 
            ylabel = "Accuracy");       
    ax_acc.legend()    
