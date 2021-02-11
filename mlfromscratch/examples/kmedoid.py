from mlfromscratch.unsupervised_learning import k_medoid
import sys
import os
import math
import random
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Import helper functions
from mlfromscratch.utils import Plot

def main():
    # Load the dataset
    X , y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)

    # Cluster the data using DBSCAN
    
    centers, members, costs, tot_cost, dist_mat = k_medoid.kmedoids(
        X, y, 5, max_iter=300, tol=0.00001)


    # Visualize Results
    plt.scatter(X[:, 0], y, c=members, s=50, cmap='viridis')


if __name__ == "__main__":
    main()
