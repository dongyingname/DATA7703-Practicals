# %%
import scipy.io
import numpy as np

# %%
data = scipy.io.loadmat('heightWeight.mat')
data = data['heightWeightData']
data_xy = data[:,1:]

# %%
data_xy

# %%
def naiveKmeans(data, k, epsilon = 0.1):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    assert len(data.shape) == 2
    centroids = np.random.uniform(data_min, data_max,size=(k,data.shape[1]))
    while 1: 
        dists = []
        # Loop to create a vector of euclidean distances
        for center in range(k):
            # Euclidean distance
            distances = np.linalg.norm(data-centroids[center,:], axis=1)
            dists.append(distances)
        dists = np.array(dists)
    
        # Return the indices of smallest euclidead distance
        clusters = np.argmin(dists, axis=0)
        # print(clusters)
        diffs = []
        for center in range(k):
            new_centroid = np.mean(data_xy[clusters==center],axis=0)
            diff = new_centroid - centroids[center,:]
            centroids[center,:] = new_centroid
            diffs.append(diff)
        diffs = np.array(diffs)
        if (diffs < epsilon).all():
            return centroids

# %%
means = naiveKmeans(data_xy,2, epsilon=0.001)
labels= data[:,0]
# data1 = data_xy[labels==1]
# data2 = data_xy[labels==2]


# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

km = KMeans(2).fit(data_xy)

plt.scatter(data1[:,0],data1[:,1], c='r', alpha=0.4)
plt.scatter(data2[:,0],data2[:,1], c='b', alpha=0.4)
plt.scatter(means[:,0],means[:,1], marker='x',c='k')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], marker='v', c='g')