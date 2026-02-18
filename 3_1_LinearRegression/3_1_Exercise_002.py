
'''
3_1 Exercise 002
'''

residuals = model.results_.resid
print("First  five residuals: ", residuals[:5])

print(f"\n The sum (and therefore also the mean) of the residuals is zero: {residuals.sum():.4}" )  

print(f"\n The mean of the squared residuals is: {(residuals**2).mean():.4f}" )
X = formula_trnsf.fit_transform(df[inputs])
Y = df[output].values

print(f"\n The rsme is : {model.score(X, Y) :.4f}" )
model.results_.summary()