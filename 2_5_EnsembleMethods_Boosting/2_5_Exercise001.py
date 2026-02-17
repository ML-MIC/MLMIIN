from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# Identify the inputs by type
num_inputs = inputs
ohe_inputs = []

# Define the preprocessor

num_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_inputs),
        ("cat", "passthrough", ohe_inputs),
    ]
)

# Define the hyperparameter grid
hyp_grid = {'Dtree__max_depth': range(2, 10)} 

# Create a pipeline
DecTree_pipe = Pipeline(steps=[('preproc', preprocessor), 
                           ('Dtree', DecisionTreeClassifier(criterion='gini', 
                                                     random_state=1))]) 

num_folds = 10
# Create a GridSearchCV object
DecTree_gridCV = GridSearchCV(estimator=DecTree_pipe, 
                        param_grid=hyp_grid, 
                        cv=num_folds,
                        return_train_score=True)

# Fit the model
DecTree_gridCV.fit(XTR, YTR)
DecTree_gridCV.best_params_

# Create model identifiers
model = DecTree_gridCV
model_name = "DtreeGridSearch"


# Add predictions for the training dataset
newCol = 'Y_'+ model_name +'_prob_neg'; 
dfTR_eval[newCol] = model.predict_proba(XTR)[:, 0]
newCol = 'Y_'+ model_name +'_prob_pos'; 
dfTR_eval[newCol] = model.predict_proba(XTR)[:, 1]
newCol = 'Y_'+ model_name +'_pred'; 
dfTR_eval[newCol] = model.predict(XTR)

# Add predictions for Test dataset
newCol = 'Y_'+ model_name +'_prob_neg'; 
dfTS_eval[newCol] = model.predict_proba(XTS)[:, 0]
newCol = 'Y_'+ model_name +'_prob_pos'; 
dfTS_eval[newCol] = model.predict_proba(XTS)[:, 1]
newCol = 'Y_'+ model_name +'_pred'; 
dfTS_eval[newCol] = model.predict(XTS)

# Obtain the scores for the training and test datasets
print(f"Modelname: {model_name}\nModel score in training: {model.score(XTR, YTR)},\nModel score in test: {model.score(XTS, YTS)}")

