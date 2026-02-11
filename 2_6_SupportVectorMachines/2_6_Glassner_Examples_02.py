'''

Code adapted from "Deep Learning - A Visual Approach" by Andrew Glassner
MIT License, see:
https://github.com/blueberrymusic/Deep-Learning-A-Visual-Approach

'''


# Show some of the lines that could separate these blobs

def draw_possible_lines(X, y, MB_list):
    MB_list = [(k, u[0], u[1]) for k, u in enumerate(MB_list)]
    x_range = get_X_range(X)
    x_values = np.linspace(x_range[0], x_range[1])
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=Scatter_dot_size, cmap='cool')
    for k, slope, intercept in MB_list:
        y_values = intercept + (slope * x_values)
        plt.plot(x_values, intercept + (slope * x_values), '-k')
        ax.text(x_values[-1] + 1/3, y_values[-1], f"L{k + 1}", 
        fontsize=12, verticalalignment='bottom', 
        horizontalalignment='left')
    plt.xlim(x_range[0], x_range[1])
    plt.xticks([],[])
    plt.yticks([],[])
    #file_helper.save_figure('SVM-possible-lines')
#     plt.show()

# Hand-chosen lines that separate the blobs


y0size = (y==0).sum()
y1size = (y==1).sum()
distances_classes = [(np.sqrt((X0[0] - X1[0])**2 + (X0[1] - X1[1])**2), i, X0, j, X1) for i, X0 in enumerate(X[y ==0, :]) for j, X1 in enumerate(X[y ==1, :])]
np.argmin([d for d,_,_,_,_ in distances_classes])

closest = distances_classes[808]
midpoint = [(closest[2][0] + closest[4][0])/ 2, (closest[2][1] + closest[4][1])/ 2 ] 
midline_slope = -(closest[2][0] - closest[4][0])/ (closest[2][1] - closest[4][1])
midline_intercept = midpoint[1] - midline_slope * midpoint[0]

MB_list = [(-.05, -2), (1, -.5), (midline_slope, midline_intercept)]

draw_possible_lines(X, y, MB_list)
# plt.scatter(closest[2][0], closest[2][1], c="red", s=Scatter_dot_size, cmap='cool')
# plt.scatter(closest[4][0], closest[4][1], c="red", s=Scatter_dot_size, cmap='cool')
# plt.scatter(midpoint[0], midpoint[1], c="red", s=Scatter_dot_size, cmap='cool')
plt.axis('equal');
plt.show();
