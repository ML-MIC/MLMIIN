
dfTS = pd.DataFrame(XTS, columns=inputs)
dfTS["Y"] = YTS

# # Select the values of k
k_start = 1
k_stop = dfTR.shape[0]
k_step = 1

k_values = np.arange(start=k_start, stop=k_stop, step=k_step).astype("int")

# Create an empty list to store the accuracies
accrcies = []
accrcies_TS = []
# Loop through k values, titting models and getting accuracies
for k in k_values:
    knn_pipe = Pipeline(steps=[('scaler', StandardScaler()), 
                        ('knn', KNeighborsClassifier(n_neighbors=k))])
    knn_pipe.fit(dfTR[inputs], dfTR[output])

    accrcies.append(knn_pipe.score(dfTR[inputs], dfTR[output]))
    accrcies_TS.append(knn_pipe.score(dfTS[inputs], dfTS[output]))
    
accrcies = np.array(accrcies)
accrcies_TS = np.array(accrcies_TS)

len(k_values), accrcies.shape, accrcies_TS.shape

max(k_values)

# Plot accuracies vs k
sns.set(rc={'figure.figsize':(12, 6)})
ax_acc = sns.scatterplot(x = 1/ k_values, y = 1 - accrcies)
sns.lineplot(x = 1/ k_values, y = 1 - accrcies, ax=ax_acc, label="Train")
sns.scatterplot(x = 1/ k_values, y = 1 - accrcies_TS, ax=ax_acc)
sns.lineplot(x = 1/ k_values, y =1 - accrcies_TS, ax=ax_acc, c="red", label="Test")
# Axes labels
ax_acc.set(xlabel ="1 / k (measure of flexibility of the model)", 
           ylabel = "Error as 1 - Accuracy")