XTR_bins = XTR.copy()
XTR_bins["Y"] = YTR
nodes = np.linspace(XTR["balance"].min(), XTR["balance"].max(), 10)
centers = [(nodes[i+1] + nodes[i])/2 for i in range(len(nodes) - 1)]
XTR_bins["bin"] = pd.cut(XTR_bins["balance"], 
                         bins=nodes)
bin_probs = XTR_bins.groupby(by="bin", observed=True)["Y"].mean().values
sns.set(rc={'figure.figsize':(10, 3)});
sns.scatterplot(XTR, x = "balance", y=YTR, hue=YTR);
sns.scatterplot(x = centers, y=bin_probs, s = 100, color = "green");