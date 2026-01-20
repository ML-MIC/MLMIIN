
'''
Exercise in 2_1_Classif_EDA_Preprocessing
'''

# Use this function with the numeric inputs in `XTR`. 

out_XTR = explore_outliers(XTR, num_inputs)
out_XTR

# Then use the result to drop all the outliers. 
# Make sure that you actually remove them from `XTR`. 

out_XTR_indices = set([k for var in num_inputs for k in out_XTR[var]["indices"] ])
XTR.drop(out_XTR_indices, axis=0, inplace=True)

# Also make sure to remove the corresponding output values in `YTR`.

YTR.drop(out_XTR_indices, axis=0, inplace=True) # Always keep in mind that you need to keep `YTR` updated.

explore_outliers(XTR, num_inputs)