# %load "./3_3_ResidualPlots.py"

def ResidualPlots(model=None, resid=[], fitted=[], X = None, Y = None, num_inputs=[], cat_inputs=[], output="Y"):

    if(len(resid) == 0) or (len(fitted) == 0):
        fitted = model.predict(X)    
        resid = Y - fitted
        
    # Handle GridSearchCV by getting best_estimator_
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_

    try:
        # Check if the model has a preprocessor in its named_steps
        if 'preprocessor' in model.named_steps:
            data = model.named_steps['preprocessor'].fit_transform(X)
        else:
            raise AttributeError("The model does not have a 'preprocessor' in its named_steps.")
    except AttributeError as e:
        warnings.warn(str(e))
        return

	
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
        print("No numerical inputs provided.")

    if len(num_inputs) > 0:

        fig, axes = plt.subplots(nrows=len(num_inputs), ncols=1, figsize=(3, 3 * len(num_inputs)), sharey=False, sharex=False)
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
        print("No categorical inputs provided.")
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
