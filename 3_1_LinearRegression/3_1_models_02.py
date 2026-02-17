# %load  "3_1_models_02.py"

sns.set(rc={'figure.figsize':(5, 5)})

df = pd.read_csv("./3_1_simple_linear_regression_01.csv")

inputs =["X0"]
output = "Y"
model_Formula = " + ".join(inputs)
formula_trnsf = FormulaTransformer(formula=model_Formula)
formula_trnsf
formula_trnsf.fit_transform(df[inputs])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipeline = Pipeline([
    ("formula", FormulaTransformer(model_Formula)),
    ("regressor", StatsModelsRegressor(OLS, fit_intercept = False))])

pipeline.fit(df[inputs], df[output])

model = pipeline._final_estimator


model.coef_
b0 = model.coef_.iloc[0]
b1 = model.coef_.iloc[1]

X_new = np.linspace(df[inputs].min(), df[inputs].max(), 10).ravel()
Y_new = pipeline.predict(pd.DataFrame({"X0":  X_new}))

plt.plot(df[inputs], df[output], 'bo')
plt.plot(X_new, Y_new, 'r-')
plt.title("The regression line is y = {:.4} + {:.4} x".format(b0, b1))
plt.show();plt.close()