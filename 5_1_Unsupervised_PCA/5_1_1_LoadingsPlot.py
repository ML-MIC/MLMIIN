# Get the loadings of x and y axes
xs = loadings['PC1']
ys = loadings['PC2']
 
# Plot the loadings on a scatterplot
for i, varnames in enumerate(feature_names):
    plt.scatter(xs.iloc[i], ys.iloc[i], s=200)
    plt.arrow(
        0, 0, # coordinates of arrow base
        xs.iloc[i], # length of the arrow along x
        ys.iloc[i], # length of the arrow along y
        color='r', 
        head_width=0
        )
    plt.text(xs.iloc[i], ys.iloc[i], varnames)
 
# Define the axes
xticks = np.linspace(-0.4, 0.8, num=5)
yticks = np.linspace(-0.4, 0.8, num=5)
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel('PC1')
plt.ylabel('PC2')
 
# Show plot
plt.title('2D Loading plot with vectors')
plt.grid()
plt.show()