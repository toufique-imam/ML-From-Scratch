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
    kmedoid_cluster = KMediod(X , 2 , 1)
    medoids , labels = kmedoid_cluster.fit(10, max_steps=1000)
    print(medoids , labels)
    p = Plot()
    p.plot_in_2d(X , labels , "Preds")
    p.plot_in_2d(X , y , "Actual")
    
if __name__ == "__main__":
    main()
