import random

import sys
import pandas as pd
import numpy as np
from scipy.spatial import distance

class KMeans:
    filepath: str
    dataset: pd.Series
    vectors: [tuple]
    dimensions: int  # Number of dimensions to K-Means Vectors
    k: int = 3  # Default clusters (K) is 3
    repetitions: int = 3  # Default repetitions of K-Means is 3
    centroids = []
    clusters = [[]]
    data_displayed = False
    run_complete = False

    def open_dataset(self, filepath: str):
        self.filepath = filepath
        self.dataset = pd.read_csv(filepath, sep='\t', header=1, index_col=0)  # Todo: Make this infer the separator, header and index
        self.dimensions = len(self.dataset.columns)  # This assumes the index column is not part of the values
        # Converts values in pd.Series into vectors
        self.vectors = list(map(lambda x: tuple(x), self.dataset.values))
        print("Dimensions: {}".format(self.dimensions))

    # Runs K-Means for the number of repetitions selected
    def run(self):
        if self.run_complete:
            self.clear()
        for i in range(self.repetitions):
            self.cluster_points()
            # Update the centroid positions
            self.update_centroid_pos()
        self.run_complete = True

    # Resets data in K Means
    def clear(self):
        self.centroids = []
        self.clusters = [[]]
        self.loop_pos = 0
        self.run_complete = False

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

        # Set model clusters
        self.clusters = clusters

    # Pick random starting positions for Centroids
    def init_centroids(self):
        for _ in range(self.k):
            random_point = random.choice(self.vectors)
            self.centroids.append(random_point)

    # Updates the Centroid for each cluster as the mean of points within the cluster
    def update_centroid_pos(self):
        for i in range(len(self.centroids)):
            self.centroids[i] = self.mean_vector(self.clusters[i])

    # Calculates Sum of Squared Errors (SSE)
    def calculate_sse(self):
        sse = 0
        for cluster in self.clusters:
            mean = self.mean_vector(cluster)
            for point in cluster:
                sse += np.linalg.norm(np.subtract(point, mean)) ** 2

        return sse

    # Mean of set of vectors (representing a cluster), new Centroid position for that cluster
    @staticmethod
    def mean_vector(s: [tuple]):
        try:
            dimensions = len(s[0])
            mean_vector = []
            # For each dimension of the vector
            for dimension in range(dimensions):
                axis_mean = sum([i[dimension] for i in s]) / len(s)
                mean_vector.append(axis_mean)

            return tuple(mean_vector)
        except IndexError:
            return None


    def update_k(self, k: int):
        if self.k != k:
            self.k = k
            self.clear()

    loop_pos = 0

    # Steps through one step of K-Means, either updating the centroids or updating the clusters
    def step(self):
        self.data_displayed = True
        if self.loop_pos == 0:
            self.cluster_points()
        elif self.loop_pos == 1:
            self.update_centroid_pos()

        self.loop_pos += 1
        if self.loop_pos > 1:
            self.loop_pos = 0


