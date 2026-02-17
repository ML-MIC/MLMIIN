# %load "./3_1_ResidualPlots.py"

def ResidualPlots(model, data, num_inputs=[], cat_inputs=[], output="Y"):

    resid = model.results_.resid
    fitted = model.results_.fittedvalues

    print("-"*50)
    print("Density Curve and QQ-plot of Residuals:", num_inputs)
    print("-"*50)

    fig1, axes = plt.subplots(figsize=(6, 3), nrows = 1, ncols = 2)
    fig1.tight_layout()

    print(axes)
    sns.kdeplot(x=resid, ax=axes[0])
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    
    sm.qqplot(resid, line='45', fit=True, dist=stats.norm, ax=axes[1])
    plt.show(); plt.close()


    print("-"*50)
    print("Fitted Values vs Residuals:")
    print("-"*50)

    fig2, axes = plt.subplots(figsize=(3, 3))
    fig2.tight_layout()

            
    sns.regplot(data=df, x=fitted, y=resid, order=2,  
                line_kws=dict(color="r"), scatter_kws=dict(alpha=0.25),
                scatter=True)
    plt.ylabel('Residuals')
    
    plt.show(); plt.close() 

    if len(num_inputs) > 0:
        print("-"*50)
        print("Numerical inputs:", num_inputs)
        print("-"*50)
    else:
        print("No numerical inputs exist.")

    

    if len(num_inputs) > 0:
        nrows = max((len(num_inputs) + 1) // 3, 1)  # Calculate the number of rows needed for three columns
#         print("nrows:", nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(9, 3 * nrows), 
                                 sharey=False, sharex=False, squeeze=False)
#         print("axes.shape:", axes.shape)
        # fig.tight_layout()

        for k in range(len(num_inputs)):
#             print(f"k={k}, num_inputs[k]={num_inputs[k]}")
#             print("data.shape:", data.shape)
#             print("row:", k // 3, "col:", k % 3)
            row = k // 3
            col = k % 3
            sns.regplot(data=data, x=num_inputs[k], y=resid, ax=axes[row, col],
                        line_kws=dict(color="r"), scatter_kws=dict(alpha=0.25),)
            axes[row, col].set_ylabel('Residuals')

        # Hide any unused subplots
        for i in range(len(num_inputs), nrows * 3):
            fig.delaxes(axes.flatten()[i])

        plt.show()

    print("-"*50)

    if len(cat_inputs) > 0:
        print("Categorical inputs:", cat_inputs)
    else:
        print("No categorical inputs exist.")
    print("-"*50)

    if len(cat_inputs) > 0:
        nrows = max((len(cat_inputs) + 1) // 3, 1)  # Calculate the number of rows needed for three columns
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(9, 3 * nrows), 
                                 sharey=False, sharex=False, squeeze=False)
        fig.tight_layout()

        for k in range(len(cat_inputs)):
            row = k // 3
            col = k % 3
            sns.boxplot(data=data, x=cat_inputs[k], y=resid, ax=axes[row, col])
            axes[row, col].set_ylabel('Residuals')
        
        # Hide any unused subplots
        for i in range(len(cat_inputs), nrows * 3):
            fig.delaxes(axes.flatten()[i])

        plt.show()