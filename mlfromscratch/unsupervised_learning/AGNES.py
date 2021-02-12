import numpy as np

class AgglomerativeClustering(object):
    """
    This is a centroid-linked agglomerative clustering. It just stores the centroid data point and the number of items
     for each cluster. 
     
    The steps of the agglomerative clustering algorithm:

    1. Define each data point as a different cluster, and the point itself as the centroid of the cluster. Also, assign
       1 to item count, and initialize a label array showing centroid indexes in the centroid array.
    2. Find the closest two centroids (minimum distance, maximum similarity).
    3. Add the higher indexed cluster to the lower indexed cluster by combining centroids with average mean and summing
       item counts.
    4. Replace all labels of the higher indexed cluster with labels of the lower indexed cluster in the labels array.
    5. If the length of the centroids array is equal to K, end the process, and return label array and centroid array.
       Else, go to 2nd step.
    """

    def __init__(self, cluster_count):
        self.cluster_count = cluster_count
        self.cluster_item_counts = np.array([])
        self.labels = np.array([])
        self.centroids = []

    def fit(self, data):
        if type(data) is not np.ndarray:
            data = np.array(data)

        self.cluster_item_counts = np.append(
            self.cluster_item_counts, np.ones(data.shape[0])
        )
        self.labels = np.append(
            self.labels, [len(self.centroids) + x for x in range(len(data))]
        )
        self.centroids += [d for d in data]

        while len(self.cluster_item_counts) != self.cluster_count:
            min_distance = np.inf
            i1, i2 = (0, 0)
            for i in range(len(self.cluster_item_counts)):
                distances = np.linalg.norm(self.centroids - self.centroids[i], axis=1)
                distances[distances == 0] = np.inf
                min_dist = np.min(distances)
                if min_dist < min_distance:
                    min_distance = min_dist
                    i1, i2 = i, np.argmin(distances)

            if i2 < i1:
                i1, i2 = i2, i1
            new_centroid = self.weigted_average(
                self.centroids[i1],
                self.centroids[i2],
                self.cluster_item_counts[i1],
                self.cluster_item_counts[i2],
            )
            self.centroids[i1] = new_centroid
            self.cluster_item_counts[i1] += self.cluster_item_counts[i2]
            self.cluster_item_counts = np.delete(self.cluster_item_counts, i2)
            del self.centroids[i2]
            self.labels[self.labels == i2] = i1
            self.labels[self.labels > i2] -= 1

        return np.array(self.centroids), self.labels

    def weigted_average(self, centroid1, centroid2, weight1, weight2):
        new_centroid = [0, 0, 0]
        for i in range(3):
            new_centroid[i] = np.average(
                [centroid1[i], centroid2[i]], weights=[weight1, weight2]
            )
        return np.array(new_centroid)
