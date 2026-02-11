'''

Code adapted from "Deep Learning - A Visual Approach" by Andrew Glassner
MIT License, see:
https://github.com/blueberrymusic/Deep-Learning-A-Visual-Approach

'''


# Find the closest point perpendicular to this line

def get_nearest_point(X, slope, intercept):
    dvals = [np.abs((intercept+(slope * x - y))/np.sqrt(1 + slope**2)) for x, y in X]
#     i = np.argmin(dvals)
    idx = np.where(dvals == np.min(dvals))[0]
#     print(idx)
    d = dvals[idx[0]]
    return (d, idx)    
    
# Draw separating lines along with their support (distance to nearest point)

def draw_possible_lines_with_support(X, y, MB_list):
    x_range = get_X_range(X)
    y_range = x_range
    x_values = np.linspace(x_range[0], x_range[1])
    plt.figure(figsize=(16, 4))
    for i in range(len(MB_list)):
        plt.subplot(1, 3, i+1)
        (slope, intercept) = MB_list[i]
        (d, idx) = get_nearest_point(X, slope, intercept)
        for id in idx:
	        plt.scatter([X[id,0]], [X[id,1]], facecolors='none', s=Scatter_dot_size*6, edgecolors='black', linewidth=2, zorder=50)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=Scatter_dot_size, cmap='cool')
        plt.plot(x_values, (intercept+(slope*x_values)), '-k')
        level = np.abs(X[idx[0],1] - slope * X[idx[0],0] - intercept)
        y_values_1 = (intercept + level + (slope*x_values))
        y_values_2 = (intercept - level + (slope*x_values))
        # intercept_a = X[i,1] - slope * X[i,0]
        plt.fill_between(x_values, y_values_1, y_values_2, edgecolor='none', color='#aaaaaa', alpha=0.4)
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        plt.xticks([],[])
        plt.yticks([],[])
        plt.title(f"L{i + 1}")
        #file_helper.save_figure('SVM-possible-lines-with-support')
    plt.axis('equal')
    plt.show()


    
# Show some hand-picked separating lines with their support

draw_possible_lines_with_support(X, y, MB_list)