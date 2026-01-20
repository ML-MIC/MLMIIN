import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': [0.1, 1, 10]}
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

train_indices = grid_search.cv_results_['split0_train_indices']
test_indices = grid_search.cv_results_['split0_test_indices']

print(f'Índices de las muestras de entrenamiento en el primer fold: {np.array(train_indices)}')
print(f'Índices de las muestras de prueba en el primer fold: {np.array(test_indices)}')
