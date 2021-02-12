from mlfromscratch.unsupervised_learning.AGNES import AgglomerativeClustering as AGNES
from sklearn import datasets
import numpy as np

# Import helper functions
from mlfromscratch.utils import Plot


def main():
    # Load the dataset
    X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)
    print(X.shape)
    # Cluster the data using Agnes
    agnes_cluster = AGNES(10)
    centroids, labels = agnes_cluster.fit(X)
    p = Plot()
    p.plot_in_2d(X, labels, "Preds")
    p.plot_in_2d(X, y, "Actual")


if __name__ == "__main__":
    main()
