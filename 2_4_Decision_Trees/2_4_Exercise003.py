# The sorted fataset is obtained.
dfTR = XTR.copy()
dfTR["Y"] = YTR
dfTR.sort_values(by="balance", ascending=True, inplace=True)

# The initial value of purity is:
print("Initial Gini = ", 2 * np.prod(dfTR["Y"].value_counts(normalize=True)))
