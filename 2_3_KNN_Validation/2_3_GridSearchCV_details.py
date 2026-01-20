'''
2_3_GridSearchCV_details.py
'''

from sklearn.model_selection import cross_validate, StratifiedKFold

# At end best_score is the best test score across the hyperparameter grid 
best_score = 0
# A list to store information about cross validation in the folds
keep_cv = list()

# Loop through the grid
for k in k_values:

    # for each combination of parameters, train a model that uses them
    knn_pipe = Pipeline(steps=[('preproc', preprocessor), 
                           ('knn', KNeighborsClassifier(k))])
    # perform cross-validation

	# to be explained in the session
    # cv_splitter = StratifiedKFold(shuffle=True, n_splits=num_folds, random_state=k)
    knn_cv = cross_validate(knn_pipe, XTR, YTR, cv=cv_splitter, return_indices=True, return_train_score=True)

    # store cv information
    keep_cv.append(knn_cv)

    # compute mean cross-validation accuracy for that point in the grid
    score = np.mean(knn_cv["test_score"])
    
    # if the new score id better, keep it and also the hyperparameters
    if score > best_score:
        best_score = score
        best_parameters = {'k': k}
        
# rebuild a model on the combined training and validation set
knn_final = Pipeline(steps=[('preproc', preprocessor), 
                           ('knn', KNeighborsClassifier(best_parameters['k']))])
knn_final.fit(XTR, YTR)

# Print Results
print("Selected as Best parameter k = ", best_parameters)
print("Final train score = ", knn_final.score(XTR, YTR))
print("Final test score = ", knn_final.score(XTS, YTS))

