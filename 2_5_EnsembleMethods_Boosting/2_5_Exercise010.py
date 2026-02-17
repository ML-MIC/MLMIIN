from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ParameterGrid

n_trees = 250

# Define the original hyperparameter grid
hyp_grid = {
    'BagDT__estimator__max_depth': list(range(1, 6)),  # Ensure it's a list
    'BagDT__max_features': [0.3, 0.5, 1.0],  # Make sure 1 is a float for consistency
    'BagDT__max_samples': [0.3, 0.5, 1.0]
}

# Generate all possible parameter combinations
all_params = list(ParameterGrid(hyp_grid))

# Filter out invalid combinations
valid_params = [
    params for params in all_params
    if not (params["BagDT__max_features"] == 1.0 and params["BagDT__max_samples"] == 1.0)
]

# Convert back to a dictionary format that GridSearchCV expects
param_grid = {
    "BagDT__estimator__max_depth": sorted(set(p["BagDT__estimator__max_depth"] for p in valid_params)),
    "BagDT__max_features": sorted(set(p["BagDT__max_features"] for p in valid_params)),
    "BagDT__max_samples": sorted(set(p["BagDT__max_samples"] for p in valid_params)),
}

# Define the BaggingClassifier
BagDT = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    oob_score=True,
    n_estimators=n_trees,
    random_state=1
)

# Define the pipeline
BagDT_pipe = Pipeline(steps=[('BagDT', BagDT)]) 

# Set up GridSearchCV
num_folds = 10
BagDT_gridCV = GridSearchCV(
    estimator=BagDT_pipe, 
    param_grid=param_grid, 
    cv=num_folds,
    return_train_score=True,
    n_jobs=-1
)

# Fit the model
BagDT_gridCV.fit(XTR, YTR)

BagDT_gridCV.score(XTR, YTR), BagDT_gridCV.score(XTS, YTS)

model_name = "BagDT"
model = BagDT_gridCV
modelDict[model_name] = {"model" : model, 
                         "inputs":inputs,
                         "scores" : {"train" : BagDT_gridCV.score(XTR, YTR), "test" : BagDT_gridCV.score(XTS, YTS)}}

print("The selected final model uses this hyperparameter values:")
print(BagDT_gridCV.best_params_)

print("and this are the scores in training and test:")
print(f"Train: {BagDT_gridCV.score(XTR, YTR):.{4}f}")   
print(f"Test: {BagDT_gridCV.score(XTS, YTS):.{4}f}")

# Now let us get the variable importances. Extract the best model
best_model = BagDT_gridCV.best_estimator_

# Access the BaggingClassifier inside the pipeline
bagging_clf = best_model.named_steps["BagDT"]

# Get feature importances from all trees
tree_importances = np.array([tree.feature_importances_ for tree in bagging_clf.estimators_])

# Compute mean feature importance across all trees
mean_importances = np.mean(tree_importances, axis=0)

# Create a DataFrame for better visualization
var_importances = pd.DataFrame({
    "variable": XTR.columns,  # Adjust based on your feature names
    "importance": mean_importances
})

# Sort features by importance
var_importances = var_importances.sort_values(by="importance", ascending=False)


var_importances['rel_importance'] = var_importances['importance'].cumsum() / var_importances['importance'].sum()
over75_idx = var_importances[var_importances['rel_importance'] >= 0.75].index[0]
var_importances.loc[:over75_idx,:]	

# To get the model performance analysis, copy the line below to a new cell, uncomment ans run it.
# %run -i "../2_4_Decision_Trees/2_4_Performance_Analysis_Binary_Classifier.py"