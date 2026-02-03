print("Fitted Tree for Model "+ model_name)
if( type(model.best_estimator_.named_steps["Dtree"]) == type((DecisionTreeClassifier()))):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(60, 60))
    plot_tree(model.best_estimator_.named_steps["Dtree"], filled=True)
    plt.show()