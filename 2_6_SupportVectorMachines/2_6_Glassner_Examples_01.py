'''

Code adapted from "Deep Learning - A Visual Approach" by Andrew Glassner
MIT License, see:
https://github.com/blueberrymusic/Deep-Learning-A-Visual-Approach

'''


# Make a File_Helper for saving and loading files.

save_files = False

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(current_dir)) # path to parent dir
from DLBasics_Utilities import File_Helper
file_helper = File_Helper(save_files)
# Create samples from a blob in this box

def make_blob_in_box(num_samples, random_state, xlo, xhi, ylo, yhi, cluster_std=1):
    xy, c = make_blobs(n_samples=num_samples, centers=1, random_state=random_state, cluster_std=cluster_std)
    x_min = min(xy[:,0])
    x_max = max(xy[:,0])
    y_min = min(xy[:,1])
    y_max = max(xy[:,1])
    xy = [np.array([np.interp(v[0], [x_min, x_max], [xlo, xhi]), 
                    np.interp(v[1], [y_min, y_max], [ylo, yhi])]) for v in xy]
    return np.array(xy)
# Make some distinct blobs

def make_demo_blobs_distinct():
    blob_samples = 50
    v1 = make_blob_in_box(num_samples=blob_samples, random_state=1, xlo= -6, xhi= 0, ylo=1, yhi=5)
    v2 = make_blob_in_box(num_samples=blob_samples, random_state=2, xlo= 0, xhi = 6, ylo=-8, yhi=-3)
    c1 = [0]*len(v1)
    c2 = [1]*len(v2)
    allv = np.append(v1, v2, axis=0)
    allc = np.append(c1, c2, axis=0)
    return (allv, allc)


# Draw the blobs as a scatter plot

def draw_demo_blobs(X, y, filename):
    x_range = get_X_range(X)
    plt.xlim(x_range[0], x_range[1])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=Scatter_dot_size, cmap='cool')
    plt.xticks([],[])
    plt.yticks([],[])
    file_helper.save_figure(filename)
    plt.axis('equal')
    plt.show()

# Convenience to get the range of X values

def get_X_range(X):
    return [min(X[:, 0])*1.1, max(X[:, 0])*1.1]

# Show the starting distinct blobs

Scatter_dot_size = 50

X, y = make_demo_blobs_distinct()  
draw_demo_blobs(X, y, 'SVM-demo-blobs-distinct')

