
df2 = df.sample(n = 1200, random_state=2)

X2 = df2.drop(columns=output)
Y2 = (df2[output] == "Yes") * 1.0

XTR2, XTS2, YTR2, YTS2 = train_test_split(X2, Y2,
                                      test_size=0.2,  
                                      random_state=2, 
                                      stratify = Y)   
                                      
LogReg_pipe2 = Pipeline(steps=[('scaler',StandardScaler()), # Preprocess the variables when training the model 
                        ('LogReg', LogisticRegression(penalty=None))]) # Model to use in the pipeline

_ = LogReg_pipe2.set_output(transform="pandas")                                      

LogReg_pipe2.fit(XTR2, YTR2);

LogReg_coeff2 = np.hstack((LogReg_pipe2.named_steps["LogReg"].intercept_[np.newaxis, :], LogReg_pipe2.named_steps["LogReg"].coef_))
b0_2, b1_2 = LogReg_coeff2.ravel()

XTR_transf2 = StandardScaler().fit_transform(XTR2)
XTR_x2 = XTR_transf2.ravel()

sns.scatterplot(x = XTR_x, y=YTR, hue=YTR)
x = np.linspace(min(XTR_x) - 0.5, max(XTR_x) + 0.5, 500)
y = 1 / (1 + np.exp(-b0 - b1 * x))

x2 = np.linspace(min(XTR_x2) - 0.5, max(XTR_x2) + 0.5, 500)
y2 = 1 / (1 + np.exp(-b0_2 - b1_2 * x2))

plt.plot(x, y, 'r-')
plt.plot(x2, y2, 'b-')



