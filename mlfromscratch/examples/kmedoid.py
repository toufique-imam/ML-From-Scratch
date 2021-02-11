from mlfromscratch.unsupervised_learning import kmedoid
from mlfromscratch.unsupervised_learning.kmedoid import KMediod
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
    kmedoid_cluster = KMediod(X , 5 , 1)
    medoids , labels = kmedoid_cluster.fit(2, max_steps=1000)
    

if __name__ == "__main__":
    main()
