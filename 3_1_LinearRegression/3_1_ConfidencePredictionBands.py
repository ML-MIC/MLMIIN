
# # Set the random seed
# rng = np.random.default_rng(1)

# Set the figure size
sns.set(rc={'figure.figsize':(8, 8)})

# # Common error variance for the linear model
# sigma = 0.5
# # Number of samples
# N = 5
# # Sample size
# n = 80
# # A column to distinguish the data in each sample from the other samples
# sampleId = np.repeat(np.arange(N), n)
# # The X part of the samples
# X = rng.uniform(size = N * n)
# # The error variables
# Eps = rng.normal(loc = 0, scale = sigma, size = N * n)
# # And the Y part according to the model
# beta0 = 4
# beta1 = -2
# Y = beta0 + beta1 * X + Eps
# # Put it all together in a DataFrame
# df = pd.DataFrame({'X': X, 'Y':Y, 'sampleId':sampleId})
# #print(df.head(50)) # test the result



# # And plot it using color to identify samples
# showLegend = True
# if(N > 10):
#     showLegend = False
# sns.scatterplot(data = df, x = X, y = Y, hue = sampleId, 
#                 palette="deep", alpha = min(1, 10/n),             
#                 legend= showLegend)

# This function gets the coefficients for the regression line
# of Y vs X. Both are assumed t be numerical pandas series of the
# same length.
def getLM(X, Y):
    modelXY = LinearRegression(fit_intercept=True)
    X = X.values[:, np.newaxis]
    Y = Y.values
    XY_fit = modelXY.fit(X, Y)
    b1 = XY_fit.coef_[0]
    b0 = XY_fit.intercept_
    return((b0, b1))

# Now let us fit a regression line for each sample and plot the result.
palette2 = iter(sns.color_palette(palette="deep", n_colors=N))


X_1 = sm.add_constant(X)
X_1
model_df0 = sm.OLS(Y, X_1) # OLS comes from Ordinary Least Squares
df0_fit = model_df0.fit()

X_new = np.linspace(X.min(), X.max(), 100)
X_new = sm.add_constant(X_new)
X_new[:5, :]

df0_fit.pred = df0_fit.get_prediction(X_new)
df0_fit.pred = df0_fit.get_prediction(X_new)

df0_fit_fitted_new = df0_fit.pred.summary_frame(alpha=0.05)["mean"]
df0_fit_confBand_low = df0_fit.pred.summary_frame(alpha=0.05)["mean_ci_lower"]
df0_fit_confBand_high = df0_fit.pred.summary_frame(alpha=0.05)["mean_ci_upper"]





############################################################

fig, ax = plt.subplots(figsize=(8, 6))


# Now let us fit a regression line for each sample and plot the result.
palette2 = iter(sns.color_palette(palette="deep", n_colors=N))
for sample in range(N):
    # select the sample
    dfs = df.loc[sampleId == sample, :] 
    Xs = dfs.X
    Ys = dfs.Y
    # fit the regression line
    b0, b1 = getLM(Xs,Ys)
    #print(sample, b0, b1,"\n", "--"*5)
    # plot the line
    Xnew = np.linspace(0, 1, num = 100)
    Ynew = b0 + b1 * Xnew

df0_fit_predBand_low = df0_fit.pred.summary_frame(alpha=0.05)["obs_ci_lower"]
df0_fit_predBand_high = df0_fit.pred.summary_frame(alpha=0.05)["obs_ci_upper"]

ax.plot(X, Y, "o", label="data")

ax.plot(X_new[:, 1], df0_fit_predBand_low, "y--", lw = 4)
ax.plot(X_new[:, 1], df0_fit_predBand_high, "y--", lw = 4)
ax.fill_between(X_new[:, 1], 
                y1 = df0_fit_predBand_low, 
                y2 = df0_fit_predBand_high, 
                color='cyan', alpha = 0.75)


ax.plot(X_new[:,1], df0_fit_confBand_low, "y--", lw = 4)
ax.plot(X_new[:,1], df0_fit_confBand_high, "y--", lw = 4)
ax.fill_between(X_new[:,1], 
                y1 = df0_fit_confBand_low, 
                y2 = df0_fit_confBand_high, 
                color='yellow', alpha = 1)
