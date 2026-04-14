import numpy as np
import matplotlib.pyplot as plt

blob1 = np.random.multivariate_normal(mean=[-3, 3], cov=[[3, 2], [2, 3]], size=100)
blob2 = np.random.multivariate_normal(mean=[5, 2], cov=[[2, 1], [1, 2]], size=100)
blob3 = np.random.multivariate_normal(mean=[0, -3], cov=[[2, 0], [0, 2]], size=100)
data = np.vstack([blob1, blob2, blob3])

plt.scatter(data[:, 0], data[:, 1])