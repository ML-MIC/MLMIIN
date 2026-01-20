from matplotlib.gridspec import GridSpec

clf_list = [(LogReg_pipe, "Logistic Regression")]
fig = plt.figure(figsize=(15, 3))
gs = GridSpec(1, 2)

plt.subplots_adjust(left=1.2, bottom=0.4, right=1.5, top=1, wspace=1, hspace=0.1)
grid_positions = [(0, 0), (0, 1)]

row, col = [0, 0]
ax = fig.add_subplot(gs[row, col])

ax.hist(
    dfTR_eval.loc[(dfTR_eval["Y"] == 0)]["Y_LR_prob_pos"],
    range=(0, 1),
    bins=10,
    # label= f'Prediction Y = 0',
    color="firebrick",
)
ax.set(title=f'True value Y = 0', xlabel="Mean predicted probability", ylabel="Count")

row, col = [0, 1]
ax = fig.add_subplot(gs[row, col])

ax.hist(
    dfTR_eval.loc[(dfTR_eval["Y"] == 1)]["Y_LR_prob_pos"],
    range=(0, 1),
    bins=10,
    # label= f'Prediction Y = 0',
    color="mediumblue",
)

ax.set(title=f'True Value Y = 1', xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()  


