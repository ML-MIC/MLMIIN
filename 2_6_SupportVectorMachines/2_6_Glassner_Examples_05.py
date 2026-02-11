'''

Code adapted from "Deep Learning - A Visual Approach" by Andrew Glassner
MIT License, see:
https://github.com/blueberrymusic/Deep-Learning-A-Visual-Approach

'''


# Make some overlapping blobs

def make_demo_blobs_overlap():
    blob_samples = 50
    v1 = make_blob_in_box(num_samples=blob_samples, random_state=1, cluster_std=0.05,
                          xlo= -2, xhi = 2, ylo = -1, yhi = 3)
    v2 = make_blob_in_box(num_samples=blob_samples, random_state=2, cluster_std=0.05,
                          xlo= -1, xhi = 3, ylo= -3, yhi= 1)
    c1 = [0]*len(v1)
    c2 = [1]*len(v2)
    allv = np.append(v1, v2, axis=0)
    allc = np.append(c1, c2, axis=0)
    return (allv, allc)

# Show the overlapping blobs

X_overlap, y_overlap = make_demo_blobs_overlap()  
draw_demo_blobs(X_overlap, y_overlap, 'SVM-demo-blobs-overlap')