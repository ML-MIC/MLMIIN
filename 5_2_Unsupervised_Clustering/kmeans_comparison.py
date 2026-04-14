from timeit import timeit
from sklearn.cluster import MiniBatchKMeans

K = 50
times = np.empty((K, 2))
inertias = np.empty((K, 2))
for k in range(1, K + 1):
    kmeans_ = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    print("\r{}/{}".format(k, 100), end="")
    times[k - 1, 0] = timeit("kmeans_.fit(X)", number=10, globals=globals())
    times[k - 1, 1] = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
    inertias[k - 1, 0] = kmeans_.inertia_
    inertias[k - 1, 1] = minibatch_kmeans.inertia_

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(range(1, K + 1), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, K + 1), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Distortion", fontsize=14)
plt.legend(fontsize=14)
# plt.axis([1, K, 0, K])

plt.subplot(122)
plt.plot(range(1, K + 1), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, K + 1), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Training time (seconds)", fontsize=14)
# plt.axis([1, K, 0, 6])

plt.tight_layout()
plt.show()