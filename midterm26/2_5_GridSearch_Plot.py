
# We create a dataframe to store the results of cross validation in the grid search

gridplot_df = pd.DataFrame.from_dict(model.cv_results_["params"])

if(gridplot_df.shape[1] != 2):
    print("This code only works for exactly two hyperparameters!")
else:
    for fld in range(num_folds):
        scores = model.cv_results_["split" + str(fld) + "_test_score"]
        gridplot_df["fld" + str(fld)] = scores
        gridplot_df["mins"] = gridplot_df.filter(like="fld").min(axis=1)
        gridplot_df["max"] = gridplot_df.filter(like="fld").max(axis=1)
        gridplot_df["mean_test_score"] = model.cv_results_["mean_test_score"]
        gridplot_df["std_test_score"] = model.cv_results_["std_test_score"]
        gridplot_df["mean_train_score"] = model.cv_results_["mean_train_score"]
        gridplot_df["std_train_score"] = model.cv_results_["std_train_score"]
    
    # And plot them using the first hyperparameter in one axis and the other for color
    params = list(model.best_params_.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    sns.lineplot(data=gridplot_df, hue=params[-1], x=params[0], y="mean_train_score", ax=axes[0])
    sns.scatterplot(data=gridplot_df, hue=params[-1], x=params[0], y="mean_train_score", legend=False, ax=axes[0])
    axes[0].set_title(f"Grid Search Results for {model_name} in the Training Set")
    axes[0].scatter(x=model.best_params_[params[0]], y=gridplot_df["mean_train_score"].max(), s=100, c="red")
    
    # Set x-axis ticks to be integers equal to the hyperparameter values
    axes[0].set_xticks(gridplot_df[params[0]])
    
    sns.lineplot(data=gridplot_df, hue=params[-1], x=params[0], y="mean_test_score", ax=axes[1])
    sns.scatterplot(data=gridplot_df, hue=params[-1], x=params[0], y="mean_test_score", legend=False, ax=axes[1])
    axes[1].set_title(f"Grid Search Results for {model_name} in the Test Set")
    axes[1].scatter(x=model.best_params_[params[0]], y=gridplot_df["mean_test_score"].max(), s=100, c="red")

    
    # Set x-axis ticks to be integers equal to the hyperparameter values
    axes[0].set_xticks(gridplot_df[params[0]])
    plt.show();plt.close()
