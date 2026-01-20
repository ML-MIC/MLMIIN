test_fold_scores = [knn_gridCV.cv_results_[f"split{k}_test_score"] for k in range(num_folds)]
test_fold_scores = np.array(test_fold_scores)
test_fold_scores = pd.DataFrame(test_fold_scores, columns= ['k_' + str(_) for _ in param_values])

ymin = test_fold_scores.min().values
ymax = test_fold_scores.max().values

param_values = knn_gridCV.cv_results_['param_knn__n_neighbors']

fig, ax = plt.subplots()

mean_train_scores = knn_gridCV.cv_results_['mean_train_score']
plt.plot(param_values, mean_train_scores, marker='o', label='Mean Train Score')

mean_test_scores = knn_gridCV.cv_results_['mean_test_score']
plt.plot(param_values, mean_test_scores, marker='*', label='Mean Test Score')

ax.vlines(param_values, ymin=ymin, ymax=ymax, colors='r', linestyles='solid', lw=2)

plt.xlabel('k')
plt.ylabel('Score (accuracy)')
plt.title('Hyperparameter Tuning Results')
plt.legend()  # Add a legend to the plot
plt.show()
plt.close()


