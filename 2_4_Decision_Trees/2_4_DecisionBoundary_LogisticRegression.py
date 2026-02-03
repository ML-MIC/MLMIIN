model = modelDict[model_name]["model"]

from matplotlib.gridspec import GridSpec
from sklearn.inspection import DecisionBoundaryDisplay

fig = plt.figure(figsize=(20, 6))
gs = GridSpec(1, 2)

plt.subplots_adjust(wspace=0.25)

row, col = [0, 0]
ax0 = fig.add_subplot(gs[row, col])
                                            
row, col = [0, 1]
ax1 = fig.add_subplot(gs[row, col])

                                                                            
X1g, X2g = np.meshgrid(np.linspace(dfTR.X1.min()*1.5, dfTR_eval.X1.max()*1.5, 100), 
                       np.linspace(dfTR.X2.min()*1.5, dfTR_eval.X2.max()*1.5, 100))                      

X1_X2 = pd.DataFrame(np.vstack([X1g.ravel(), X2g.ravel()]).transpose(), columns=["X1", "X2"])

X1_X2["X1Sq"] = X1_X2["X1"]**2

pred_class = model.predict(X1_X2).reshape([100, 100])

DB = DecisionBoundaryDisplay(xx0=X1g, xx1=X2g, response=pred_class)
DB.plot(ax=ax0,  cmap="Greens_r", alpha=0.2)
sns.scatterplot(dfTR, x = "X1", y = "X2", hue="Y", palette="pastel", ax=ax0);   

pred_probs = model.predict_proba(X1_X2)[:, 1].reshape([100, 100])
uniq_probs, counts_probs = np.unique(pred_probs, return_counts=True)
                                       
contours = plt.contour(X1g, X2g, pred_probs, 
                       levels=np.linspace(0, 1, num=5), 
                       colors="black", linewidths=0.5)         
plt.clabel(contours)    

sns.scatterplot(dfTR, x = "X1", y = "X2", hue="Y", palette="pastel", ax=ax1);   

