def gini_subnodes(df, var, split):
    num_l = (df[var] <= split).sum()
    n = df.shape[0]
    df_l = df.loc[df[var] <= split]
    df_r = df.loc[df[var] > split] 
    prop_l = df_l["Y"].value_counts(normalize=True)
    prop_r = df_r["Y"].value_counts(normalize=True)
    if (prop_l.shape[0] == 1):
        gini_l = 0
    else: 
        gini_l = 2 * np.prod(prop_l)
    if (prop_r.shape[0] == 1):
        gini_r = 0
    else: 
        gini_r = 2 * np.prod(prop_r)
    return(gini_l, gini_r, num_l, n - num_l)

