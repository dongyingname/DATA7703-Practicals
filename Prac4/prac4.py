# %%
import scipy.io
import numpy as np

# %%
data = scipy.io.loadmat('heightWeight.mat')
data = data['heightWeightData']
data_xy = data[:,1:]

# %%
def naiveKmeans(data, k, epsilon = 0.1):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    assert len(data.shape) == 2
    centroids = np.random.uniform(data_min, data_max,size=(k,data.shape[1]))
    while 1: 
        dists = []
        for center in range(k):
            distances = np.linalg.norm(data-centroids[center,:], axis=1)
            dists.append(distances)
        dists = np.array(dists)
        clusters = np.argmin(dists, axis=0)
        diffs = []
        for center in range(k):
            new_centroid = np.mean(data_xy[clusters==center],axis=0)
            diff = new_centroid - centroids[center,:]
            centroids[center,:] = new_centroid
            diffs.append(diff)
        diffs = np.array(diffs)
        if (diffs < epsilon).all():
            return centroids