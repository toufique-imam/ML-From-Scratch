from numpy.random import choice
from numpy.random import seed
import numpy as np


class KMediod():
    def __init__(self, X=None, k=None, seed_value=1):
        seed(seed_value)
        self.datapoints = X
        self.K = k

    def init_medoids(self):
        samples = choice(len(self.datapoints), size=self.K, replace=False)
        self.medoids_initial = self.datapoints[samples, :]

    def compute_distance(X, mediods, p):
        m = len(X)
        mediods_shape = mediods.shape

        if(len(mediods_shape) == 1):
            mediods = mediods.reshape((1, len(mediods)))

        k = len(mediods)

        S = np.empty((m, k))

        for i in range(m):
            d_i = np.linalg.norm(X[i, :]-mediods, ord=p, axis=1)
            S[i, :] = d_i*p

        return S

    def assign_labels(S):
        return np.argmin(S, axis=1)

    def update_medoids(self, medoids, p):
        S = self.compute_distance(self.datapoints, medoids, p)
        labels = self.assign_labels(S)
        out_medoids = medoids

        for i in set(labels):
            avg_dissimilarity = np.sum(
                self.compute_distance(self.datapoints, medoids[i], p))
            cluster_points = self.datapoints[labels == i]

            for datap in cluster_points:
                new_dissimilarity = np.sum(
                    self.compute_distance(self.datapoints, datap, p))
                if new_dissimilarity < avg_dissimilarity:
                    avg_dissimilarity = new_dissimilarity
                    out_medoids[i] = datap

        return out_medoids

    def has_converged(old_medoids, medoids):
        return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])

    def fit(self, p, X=None, k=None, starting_medoids=None, max_steps=np.inf, stopping_steps=5):
        if(X is not None):
            self.datapoints = X
        if k is not None:
            self.K = k

        if(self.datapoints is None):
            raise ValueError("No data Provided")
        if(self.K is None):
            raise ValueError("No K provided")

        if starting_medoids is not None:
            self.medoids_initial = starting_medoids
        else:
            self.init_medoids()

        convereged = False
        k = self.K

        self.labels = np.zeros(len(X))
        self.medoids_current = self.medoids_initial
        i = 1
        same_medoids = 0

        while (same_medoids < stopping_steps) and (i <= max_steps):
            old_medoids = self.medoids_current.copy()
            S = self.compute_distance(self.datapoints, self.medoids_current, p)

            self.labels = self.assign_labels(S)

            self.medoids_current = self.update_medoids(self.medoids_current, p)

            convereged = self.has_converged(old_medoids, self.medoids_current)

            if(convereged):
                same_medoids = same_medoids+1
            else:
                same_medoids = 0

        return (self.medoids_current, self.labels)
