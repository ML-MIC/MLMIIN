
dfTR_eval['Y_RND_pred_prob_pos'] = np.random.default_rng(seed=2024).uniform(size=XTR.shape[0])
dfTR_eval['Y_RND_pred_thrs'] = (dfTR_eval["Y_RND_pred_prob_pos"] >= thrs) * 1

fpr_rnd, tpr_rnd, thresholds_rnd = roc_curve(YTR, dfTR_eval["Y_RND_pred_prob_pos"], pos_label=1)

RocCurveDisplay(fpr=fpr_rnd, tpr=tpr_rnd).plot()
plt.scatter(x = np.linspace(0, 1, 100),  y = np.linspace(0, 1, 100), s=2, c="red", linestyle='dashed')
plt.show()

