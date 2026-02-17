var_importances['cum_rel_importance'] = var_importances['importance'].cumsum() / var_importances['importance'].sum()
first_pos = np.where((var_importances['cum_rel_importance']) >= 0.75)[0][0]
var_importances.iloc[:first_pos+1,:]
