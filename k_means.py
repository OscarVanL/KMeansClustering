import random

import sys
import pandas as pd
from scipy.spatial import distance

class KMeans:
    filepath: str
    dataset: pd.Series
    vectors: [tuple]
    dimensions: int  # Number of dimensions to K-Means Vectors
    k: int = 3
    centroids = []
    clusters = [[]]
    data_displayed = False



    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def open_dataset(self, filepath: str):
        self.filepath = filepath
        self.dataset = pd.read_csv(filepath, sep='\t', header=1, index_col=0)
        self.dimensions = len(self.dataset.columns)
        # Converts values in pd.Series into vectors
        self.vectors = list(map(lambda x: tuple(x), self.dataset.values))
        print("Dimensions: {}".format(self.dimensions))


    def cluster_points(self):
        if not self.centroids:
            self.init_centroids()

        # Initialise K clusters
        clusters = [[] for i in range(self.k)]

        # Categorise each point into a cluster
        for point in self.vectors:
            best_cluster = None
            smallest_dist = sys.maxsize
            for index, centroid in enumerate(self.centroids):
                dist = distance.euclidean(centroid, point)
                if dist < smallest_dist:
                    best_cluster = index
                    smallest_dist = dist
            # Put the point into the closest cluster
            clusters[best_cluster].append(point)

        # Update model clusters
        self.clusters = clusters
        # Update the centroid positions
        self.update_centroid_pos()




    # Pick random starting positions for Centroids
    def init_centroids(self):
        print("Initialising {} Centroids".format(self.k))
        for _ in range(self.k):
            random_point = random.choice(self.vectors)
            self.centroids.append(random_point)
            print("Init Centroid:", random_point)




    # Updates the Centroid for each cluster as the mean of points within the cluster
    def update_centroid_pos(self):
        for i in range(len(self.centroids)):
            self.centroids[i] = self.mean_vector(self.clusters[i])

    # Mean of set of vectors (representing a cluster), new Centroid position for that cluster
    def mean_vector(self, s: [tuple]):
        dimensions = len(s[0])
        mean_vector = []
        # For each dimension of the vector
        for dimension in range(dimensions):
            axis_mean = sum([i[dimension] for i in s]) / len(s)
            mean_vector.append(axis_mean)

        return tuple(mean_vector)

    def update_k(self, k: int):
        if self.k != k:
            self.k = k
            if self.data_displayed:
                self.centroids = []
                self.clusters = [[]]
                self.init_centroids()
                self.cluster_points()

    loop_pos = 0

    def next_step(self):
        self.data_displayed = True
        if self.loop_pos == 0:
            self.cluster_points()
        elif self.loop_pos == 1:
            self.update_centroid_pos()

        self.loop_pos += 1
        if self.loop_pos > 1:
            self.loop_pos = 0


