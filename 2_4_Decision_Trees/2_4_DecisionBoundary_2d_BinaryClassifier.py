# %run -i "./2_4_DecisionBoundary_2d_BinaryClassifier.py"


from matplotlib.gridspec import GridSpec
from sklearn.inspection import DecisionBoundaryDisplay

fig = plt.figure(figsize=(20, 6))
gs = GridSpec(1, 2)

plt.subplots_adjust(wspace=0.25)

row, col = [0, 0]
ax0 = fig.add_subplot(gs[row, col])


DB = DecisionBoundaryDisplay.from_estimator(model, dfTR[model_inputs], response_method="predict", plot_method="contour", 
                                            colors="black", ax=ax0, alpha=0.1)
DB = DecisionBoundaryDisplay.from_estimator(model, dfTR[model_inputs], response_method="predict", 
                                            cmap="Greens_r", ax=ax0, alpha=0.5)
                                            
sns.scatterplot(dfTR, x = model_inputs[0], y = model_inputs[1], hue="Y", palette="pastel", ax=ax0);   
                                            
row, col = [0, 1]
ax1 = fig.add_subplot(gs[row, col])

                                                                            
X1g, X2g = np.meshgrid(np.linspace(dfTR[model_inputs[0]].min()*1.5, dfTR[model_inputs[0]].max()*1.5, 100), 
                       np.linspace(dfTR[model_inputs[1]].min()*1.5, dfTR[model_inputs[1]].max()*1.5, 100))                      

X1_X2 = pd.DataFrame(np.vstack([X1g.ravel(), X2g.ravel()]).transpose(), columns=model_inputs)

pred_probs = model.predict_proba(X1_X2)[:, 1].reshape([100, 100])
uniq_probs, counts_probs = np.unique(pred_probs, return_counts=True)
                                       
contours = plt.contour(X1g, X2g, pred_probs, 
                       levels=np.linspace(0, 1, num=len(uniq_probs)), 
                       colors="black", linewidths=0.5)         
plt.clabel(contours)    
plt.suptitle("Decision Boundary and Probability Level Curves for "+ model_name)

sns.scatterplot(dfTR, x = model_inputs[0], y = model_inputs[1], hue="Y", palette="pastel", ax=ax1);   
