def explore_outliers(df, num_vars):
    """
    Extract information about the possible outliers in the numerical inputs of a dataset df.

    Args:
        df: a pandas dataframe containing the dataset
        num_vars: a list of strings with column names of df corresponding to (a subset of
        	the) numerical features of the dataset.

    Returns:
        dict: with keys equal to num_vars and for each key the following values:
        	values: the outlier values
        	positions: their row number in df
        	indices: the index corresponding to the row numbers         	
        	
    Plots:
    	Separate boxplots for each of the variables in num_vars
    	
    """
    
    fig, axes = plt.subplots(nrows=len(num_vars), ncols=1, figsize=(7, len(num_inputs)), sharey=False, sharex=False)
    fig.tight_layout()
	# Create the return dictionary
    outliers_df = dict()
	# Loop over the numerical inputs in num_vars
    for k in range(len(num_vars)):
        var = num_vars[k]
        # Boxplot for this variable
        sns.boxplot(df, x=var , ax=axes[k])
        # Outlying values
        outliers_df[var] = boxplot_stats(df[var])[0]["fliers"]
        # Their positions
        out_pos = np.where(df[var].isin(outliers_df[var]))[0].tolist() 
        # And their indices
        out_idx = [df[var].index.tolist()[ k ] for k in out_pos]
        # Update the dictionary
        outliers_df[var] = {"values": outliers_df[var], 
                            "positions": out_pos, 
                            "indices": out_idx}
    return outliers_df



def ResidualPlots(model=None, resid=[], fitted=[], data=None, num_inputs=[], cat_inputs=[], output="Y"):
    """
    Obtain the residual plots for a regression model, given a a dataset, an output 
    variable and lists of numerical and categorical inputs in the dataset.
    
    Two situations are considered. If the residual and fitted values are available, then 
    they will be used directly in the plots. If they are not (resid or fitted arguments 
    are empty) then a model be provided. In this case the model must have a results 
    property that contains residuals and fitted values. 

    Args:
    
		data: a pandas dataframe containing the dataset
    	output: the name of the output variable.
    	num_inputs, cat_inputs: lists of column names in the dataste containing the two
    	types of inputs.
    
        resid: the residual of a model (array like)
        fitted: the fitted values of a model (array like)
        
        if any of resid or fitted are empty:
        
        model: a regression model with a results property that contains residuals and
        	fitted values.
        

    Returns:
        None
        	
    Plots:
    	- A density curve and QQ-plot of Residuals
    	- A fitted vs residuals scatterplot
    	- A values vs residuals scatterplot for each numerical input
    	- Parallel boxplots of residuals by level for each categorical input
    """

    if(len(resid) == 0) or (len(fitted) == 0):
        resid = model.results_.resid
        fitted = model.results_.fittedvalues

    print("-"*50)
    print("Density Curve and QQ-plot of Residuals:", num_inputs)
    print("-"*50)

    fig, axes = plt.subplots(figsize=(8, 4), nrows = 1, ncols = 2)
    fig.tight_layout()

    print(axes)
    sns.kdeplot(x=resid, ax=axes[0])
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    
    sm.qqplot(resid, line='45', fit=True, dist=stats.norm, ax=axes[1])
    plt.show()


    print("-"*50)
    print("Fitted Values vs Residuals:")
    print("-"*50)

    fig, axes = plt.subplots(figsize=(4, 4))
    fig.tight_layout()

            
    sns.regplot(data=data, x=fitted, y=resid, order=2,  color=".3",
                line_kws=dict(color="r"), scatter_kws=dict(alpha=0.25),
                scatter=True)
    plt.ylabel('Residuals')
    
    plt.show();

    if len(num_inputs) > 0:
        print("-"*50)
        print("Numerical inputs:", num_inputs)
        print("-"*50)
    else:
        print("No numerical inputs exist.")

    if len(num_inputs) > 0:

        fig, axes = plt.subplots(nrows=len(num_inputs), ncols=1, figsize=(4, 4 * len(num_inputs)), sharey=False, sharex=False)
        fig.tight_layout()

        for k in range(len(num_inputs)):
            if(len(num_inputs) > 1):
                sns.regplot(data=data, x=num_inputs[k], y=resid, order=2,  color=".3",
                            line_kws=dict(color="r"), scatter_kws=dict(alpha=0.25),
                            scatter=True, ax=axes[k])
                plt.ylabel('Residuals')
            else:
                sns.regplot(data=data, x=num_inputs[k], y=resid, order=2,  color=".3",
                            line_kws=dict(color="r"), scatter_kws=dict(alpha=0.25),
                            scatter=True)
                plt.ylabel('Residuals')
        
        plt.show();

    print("-"*50)
    if len(cat_inputs) > 0:
        print("Categorical inputs:", cat_inputs)
    else:
        print("No categorical inputs exist.")
    print("-"*50)

    if len(cat_inputs) > 0:
        fig, axes = plt.subplots(nrows=len(cat_inputs), ncols=1, figsize=(4, 4 * len(cat_inputs)), sharey=False, sharex=False)
        fig.tight_layout()

        for k in range(len(cat_inputs)):
            if(len(cat_inputs) > 1):
                sns.boxplot(data=data, x=cat_inputs[k], y = resid, ax=axes[k])
                plt.ylabel('Residuals')
            else:
                sns.boxplot(data=data, x=cat_inputs[k], y = resid)
                plt.ylabel('Residuals')
        
        plt.show();
        
    return None
