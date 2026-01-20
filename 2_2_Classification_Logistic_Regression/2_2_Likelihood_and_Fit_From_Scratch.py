from scipy.optimize import minimize, LinearConstraint

def loglikelihood(b, X, Y):
    b0 = b[0]
    b1 = b[1]
    # print(b0, b1)
    # print(X[Y == 1].head())
    lkld_1 = np.sum(b0 + b1 * X[Y == 1]  - np.log(1 + np.exp(b0 + b1 * X[Y == 1])))
    # print("lkld_1 = \n", lkld_1)
    lkld_0 = np.sum(-np.log(1 + np.exp(b0 + b1 * X[Y == 0])))
    # print("lkld_0 = \n", lkld_0)
    lkld = lkld_1 + lkld_0
    return  -lkld


res = minimize(
    loglikelihood,
    x0=[1, 1],
    args=(XTR_transf, YTR)
)

print("The coefficients that minimize the -log-likelihood are ", res.x)
print("Recall that the coefficients fitted by scikit-learn are ", b0, b1)

print("The resulting minimal -log-likelihood is ")
print(loglikelihood(list(res.x), XTR_transf, YTR))