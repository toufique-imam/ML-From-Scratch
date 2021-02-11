import numpy as np
import time
from scipy.spatial import distance
from copy import deepcopy

def get_initial_centers(no_clusters, no_samples):
    # return random points as initial centers
    initial_centers = []
    while len(initial_centers) < no_clusters:
        # select temporary points from the dataset and check if they have been already chosen as centers
        temp = np.random.randint(0, no_samples)
        # if chosen random points are not chosen as centers add them to the inital centers array
        if not temp in initial_centers:
            initial_centers.append(temp)
    return initial_centers

def calculate_cost(X, centers_id):
    # this function return total cost and cost of each cluster
    st = time.time()
    dist_mat = np.zeros((len(X), len(centers_id)))
    # compute distance matrix
    for j in range(len(centers_id)):
        # place jth centroid row in the center variable
        centroid = X[centers_id[j], :]
        # compute euclidean distance of each data point from the center
        for i in range(len(X)):
            # If the ith data point is the centroid itself then the distance between them is zero
            if i == centers_id[j]:
                dist_mat[i, j] = 0.
            # If the ith data point of not the centroid then compute distance
            else:
                dist_mat[i, j] = distance.euclidean(X[i, :], centroid)
    # Chose the closest centroid for each data point
    mask = np.argmin(dist_mat, axis=1)
    members = np.zeros(len(X))
    costs = np.zeros(len(centers_id))
    for i in range(len(centers_id)):
        # mem_id contains the datapoints whose closest centroid is i
        mem_id = np.where(mask == i)
        # Assign i as the centroid of these datapoints
        members[mem_id] = i
        # Compute the cost for each centroid as the summation of the distance of all the datapoints in it's cluster
        costs[i] = np.sum(dist_mat[mem_id, i])
    return members, costs, np.sum(costs), dist_mat


def kmedoids(X, n_clusters, max_iter, tol):
    '''run algorithm return centers, members, and etc.'''
    # Get initial centers
    n_samples = X.shape[0]
    n_features = X.shape[1]
    initial_centers = get_initial_centers(n_clusters, n_samples)

    print('Initial centers are ', initial_centers)
    centers = initial_centers
    members, costs, tot_cost, dist_mat = calculate_cost(X, initial_centers)
    cc, change = 0, True
    while True:
        change = False
        for i in range(n_samples):
            if not i in centers:
                for j in range(len(centers)):
                    # Consider a datapoint other than the centroid to be the new centroid
                    centers2 = deepcopy(centers)
                    centers2[j] = i
                    # Compute costs for these new list of centers
                    members2, costs2, tot_cost2, dist_mat2 = calculate_cost(
                        X, centers2)
                    # If new total cost is smaller than old total cost then update centroids
                    if tot_cost2-tot_cost < tol:
                        members, costs, tot_cost, dist_mat = members2, costs2, tot_cost2, dist_mat2
                        centers = centers2
                        change = True
                        print('Change centers to ', centers)
        if cc > max_iter:
            print(
                'Search ended because maximum number of iterations has been reached', max_iter)
            break
        # If total cost begins to increase then stop the algorithm
        if not change:
            print('Search ended because the position of the centroids remained unchanges')
            break
        cc += 1
    return centers, members, costs, tot_cost, dist_mat
