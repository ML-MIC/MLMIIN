'''

Code adapted from "Deep Learning - A Visual Approach" by Andrew Glassner
MIT License, see:
https://github.com/blueberrymusic/Deep-Learning-A-Visual-Approach

'''

# Draw the data, along with a boundary line, the support for that line, and its support vectors
from matplotlib.colors import ListedColormap

my_cmap = ListedColormap(["orange"], name='from_list', N=None)
# m = cm.ScalarMappable(norm=norm, cmap=cmap)

def plot_boundary_and_support(X, y, model, support_vectors_alpha, C, levels = [-1, 0, 1], 
                              title=False, decisionRegion=True, contours=True):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=Scatter_dot_size, cmap='cool')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_grid = np.linspace(xlim[0], xlim[1], 30)
    y_grid = np.linspace(ylim[0], ylim[1], 30)
    Y_grid, X_grid = np.meshgrid(y_grid, x_grid)
    xy_grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    P = model.decision_function(xy_grid).reshape(X_grid.shape)

    # plot decision boundary and margins
    ax.contour(X_grid, Y_grid, P, colors='k',
               levels=[-1, 0, 1], alpha=1.0,
               linestyles=['--', '-', '--'])
    if contours:
	    disp = DecisionBoundaryDisplay.from_estimator(
    	model, X, 
	    response_method="decision_function", plot_method="contour",levels=levels,
    	xlabel=model.classes_[0], ylabel=model.classes_[1],
	    alpha=0.5, cmap=my_cmap, ax=ax)
    if decisionRegion:
        disp = DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", 
                                                      cmap="Pastel2", alpha=0.5, 
                                                      xlabel=model.classes_[0], 
                                                      ylabel=model.classes_[1],
                                                      ax=ax)
    # disp.ax_.scatter(X[:, 0], X[:, 1], c=y + 1, edgecolor="k")

    sv = model.support_vectors_
    
#     plt.scatter(sv[:,0], sv[:,1], s=Scatter_dot_size*6,  edgecolors='black',
#                     facecolors="none", linewidth=2, zorder=50, 
#                     alpha=support_vectors_alpha)
    
    dcf_X = model.decision_function(X)
    # print(np.abs(dcf_X) < 1)
    sv_cond1 = (np.abs(dcf_X) < 1)
    sv1 = X[np.where(sv_cond1)[0]]
    sv_cond2 = (dcf_X * (2 * y - 1) < 0)
    sv2 = X[np.where(sv_cond2)[0]]
    plt.scatter(sv1[:,0], sv1[:,1], s=Scatter_dot_size*6,  edgecolors='black',
                    facecolors="none", linewidth=2, zorder=50, 
                    alpha=support_vectors_alpha/2)
    plt.scatter(sv2[:,0], sv2[:,1], s=Scatter_dot_size*6,  edgecolors='black',
                    facecolors="none", linewidth=2, zorder=50, 
                    alpha=support_vectors_alpha/2)



    x_range = get_X_range(X)
    plt.xlim(x_range[0], x_range[1])
    if title:
        C_string = 'C ={0:.3f}'.format(C)
        if (C > .01): 
            C_string = 'C = {0:.0e}'.format(C)
        plt.title(C_string)
#     plt.xticks([],[])
#     plt.yticks([],[])
    file_helper.save_figure('SVM-C-'+str(C)+'-with-support')
    plt.axis('equal')
    plt.show()
    return([dcf_X, sv_cond1, sv_cond2])
#     return([dcf_X])

# For this value of C, do run an SVM and plot the results

C=1E10
model = SVC(kernel='linear', C=C)
model.fit(X, y)
decf_X = plot_boundary_and_support(X, y, model, 1.0, C)



