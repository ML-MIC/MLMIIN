'''

Code adapted from "Deep Learning - A Visual Approach" by Andrew Glassner
MIT License, see:
https://github.com/blueberrymusic/Deep-Learning-A-Visual-Approach

'''

# SVM Model Fit Begins ------------------------------------------------------

hyp_grid = {'SVM_poly__C': 10.**np.arange(-6, 6, 1)}

SVM_poly_pipe = Pipeline(steps=[ ('scaler', StandardScaler()), 
                        ('SVM_poly',  SVC(kernel='poly',
                                          degree = 2,
                                          probability=True,
                                          random_state=1))]) 

num_folds = 10

SVM_poly_gridCV = GridSearchCV(estimator=SVM_poly_pipe, 
                        param_grid=hyp_grid, 
                        cv=num_folds,
                        return_train_score=True,
                        n_jobs=-1)

model = SVM_poly_gridCV
model.fit(XTR, YTR)

# SVM Model Fit Ends ------------------------------------------------------

# 3D Visualization Begins ------------------------------------------------------


# Add a Z component scaled by distance from the origin

from mpl_toolkits.mplot3d import Axes3D

def augment_with_z(X):
    xyz = []
    for xy in X:
        r = math.sqrt((xy[0]*xy[0])+(xy[1]*xy[1]))
        z = r*r
        p = [xy[0], xy[1], z]
        xyz.append(p)
    return np.array(xyz)

# Goofy matplotlib code to show a 3D plot of the data, and a dividing plane

def show_3D_data(X, y):
    plane_z = np.array([[0.35]])
    plt.figure(figsize=(18, 7.5))
    ax = plt.subplot(1, 3, 1, projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y, s=50, cmap='cool', edgecolor='black')
    ps = 2
    xx, yy = np.meshgrid(np.arange(-ps, ps), np.arange(-ps, ps))
    ax.plot_surface(xx, yy, plane_z, color='#ffff00', edgecolor='black')
    ax.view_init(0, 20)
    plt.xlim(-ps, ps)
    plt.ylim(-ps, ps)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.show()

# Show the data with Z, and the dividing plane

Xz = augment_with_z(XTR)
show_3D_data(Xz, YTR)
