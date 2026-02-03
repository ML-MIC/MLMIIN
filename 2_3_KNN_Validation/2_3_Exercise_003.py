'''
Exercise 2_3 003
'''

# Select the values of k
k_start = 2
k_stop = 25
k_step = 1

k_values = np.arange(start=k_start, stop=k_stop, step=k_step).astype("int")

# Create an empty list to store the accuracies
accrcies = []
# Loop through k values, titting models and getting accuracies
for k in k_values:
    knn_pipe = Pipeline(steps=[('scaler', StandardScaler()), 
                        ('knn', KNeighborsClassifier(n_neighbors=k))])
    knn_pipe.fit(dfTR[inputs], dfTR[output])

    accrcies.append(knn_pipe.score(dfTR[inputs], dfTR[output]))
    
accrcies = np.array(accrcies)
# Plot accuracies vs k
ax_acc = sns.scatterplot(x = k_values, y = accrcies)
sns.lineplot(x = k_values, y = accrcies, ax=ax_acc)
# Axes labeks
ax_acc.set(xlabel ="k (num. of neighbors)", 
           ylabel = "Accuracy")
           
k_select = k_values[np.argmax(accrcies)]

plt.axvline(x=k_select, linestyle="--") 
plt.text(k_select + 0.5, np.min(accrcies) ,"k = "+str(k_select))           