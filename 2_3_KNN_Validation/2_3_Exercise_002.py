print("score (pipeline method)=", knn_pipe.score(dfTR[inputs], dfTR[output]))

from sklearn.metrics import accuracy_score
print("accuracy_score =", knn_pipe.score(dfTR[inputs], dfTR[output]))