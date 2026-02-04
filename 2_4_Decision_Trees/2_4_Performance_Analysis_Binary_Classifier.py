# Performance Analysis of a Binary Classification Model



if not('dfTR' in globals()):
    dfTR = dfTR_eval.copy()
else:
    print("dfTR already exists")
if not('dfTR' in globals()):
    dfTS = dfTS_eval.copy()

## Training predictions dataset

model = modelDict[model_name]["model"]
model_inputs = modelDict[model_name]["inputs"]
print("Model %s with model inputs %s"%(model_name, str(model_inputs)) )

newCol = 'Y_'+ model_name +'_prob_neg'; 
dfTR_eval[newCol] = model.predict_proba(dfTR[model_inputs])[:, 0]
newCol = 'Y_'+ model_name +'_prob_pos'; 
dfTR_eval[newCol] = model.predict_proba(dfTR[model_inputs])[:, 1]
newCol = 'Y_'+ model_name +'_pred'; 
dfTR_eval[newCol] = model.predict(dfTR[model_inputs])
# Test predictions dataset
dfTS_eval = dfTS[model_inputs].copy()
dfTS_eval['Y'] = YTS 
newCol = 'Y_'+ model_name +'_prob_neg'; 
dfTS_eval[newCol] = model.predict_proba(dfTS[model_inputs])[:, 0]
newCol = 'Y_'+ model_name +'_prob_pos'; 
dfTS_eval[newCol] = model.predict_proba(dfTS[model_inputs])[:, 1]
newCol = 'Y_'+ model_name +'_pred'; 
dfTS_eval[newCol] = model.predict(dfTS[model_inputs])

print(f'''
        Model score in Training = %.2f
        Model score in Test = %.2f
        '''%(
            # str(model.best_params_) ,     
           model.score(dfTR[model_inputs], YTR), 
        #    model.best_score_,
           model.score(dfTS[model_inputs], YTS)))

#       '''%(str(model.best_params_) , 
        # Model score in Validation = %.2f

### Confusion Matrices


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
fig = plt.figure(constrained_layout=True, figsize=(6, 2))
spec = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(spec[0, 0]);ax1.set_title('Training'); ax1.grid(False)
ax2 = fig.add_subplot(spec[0, 2]);ax2.set_title('Test'); ax2.grid(False)
ConfusionMatrixDisplay.from_estimator(model, dfTR[model_inputs], YTR, cmap="Greens", colorbar=False, ax=ax1, labels=[1, 0])
ConfusionMatrixDisplay.from_estimator(model, dfTS[model_inputs], YTS, cmap="Greens", colorbar=False, ax=ax2, labels=[1, 0])
plt.suptitle("Confusion Matrices for "+ model_name)
plt.show(); 

### ROC Curves

from sklearn.metrics import RocCurveDisplay
fig = plt.figure(figsize=(12, 4))
spec = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(spec[0, 0]);ax1.set_title('Training')
ax2 = fig.add_subplot(spec[0, 1]);ax2.set_title('Test')
RocCurveDisplay.from_estimator(model, dfTR[model_inputs], YTR, plot_chance_level=True, ax=ax1)
RocCurveDisplay.from_estimator(model, dfTS[model_inputs], YTS, plot_chance_level=True, ax=ax2);
plt.suptitle("ROC Curves for "+ model_name)
plt.show(); 

### Calibration Curves

from sklearn.calibration import CalibrationDisplay
fig = plt.figure(figsize=(12, 4))
spec = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(spec[0, 0]);ax1.set_title('Training')
ax2 = fig.add_subplot(spec[0, 1]);ax2.set_title('Test')
CalibrationDisplay.from_estimator(model, dfTR[model_inputs], YTR, n_bins=10, ax=ax1)
CalibrationDisplay.from_estimator(model, dfTS[model_inputs], YTS, n_bins=10, ax=ax2);
plt.suptitle("Calibration Curves for "+ model_name)
plt.show(); plt.rcParams['figure.figsize']=plt.rcParamsDefault['figure.figsize']


### Probability Histograms


from matplotlib.gridspec import GridSpec

# clf_list = [(LogReg_pipe, "Logistic Regression")]
fig = plt.figure(figsize=(15, 3))
gs = GridSpec(1, 2)

plt.subplots_adjust(left=1.2, bottom=0.4, right=1.5, top=1, wspace=1, hspace=0.1)
grid_positions = [(0, 0), (0, 1)]

row, col = [0, 0]
ax = fig.add_subplot(gs[row, col])

ax.hist(
    dfTR_eval.loc[(dfTR_eval["Y"] == 0)]['Y_'+ model_name +'_prob_pos'],
    range=(0, 1),
    bins=10,
    # label= f'Prediction Y = 0',
    color="firebrick",
)
ax.set(title=f'True value Y = 0', xlabel="Mean predicted probability", ylabel="Count")

row, col = [0, 1]
ax = fig.add_subplot(gs[row, col])

ax.hist(
    dfTR_eval.loc[(dfTR_eval["Y"] == 1)]['Y_'+ model_name +'_prob_pos'],
    range=(0, 1),
    bins=10,
    # label= f'Prediction Y = 0',
    color="mediumblue",
)

ax.set(title=f'True Value Y = 1', xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.suptitle("Probability Histograms for "+ model_name)

plt.show() 
