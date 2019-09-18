import random

import pandas as pd
from scipy.spatial import distance

class KMeans:
    filepath: str
    dataset: pd.Series
    vectors: [tuple]
    dimensions: int  # Number of dimensions to K-Means Vectors
    centeroids = []

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def open_dataset(self, filepath: str):
        self.filepath = filepath
        self.dataset = pd.read_csv(filepath, sep='\t', header=1, index_col=0)
        self.dimensions = len(self.dataset.columns)
        # Converts values in pd.Series into vectors
        self.vectors = list(map(lambda x: tuple(x), self.dataset.values))
        print("Dimensions: {}".format(self.dimensions))

    def perform_kmeans(self, k: int):
        # Pick random starting positions for Centroids
        for _ in k:
            self.centeroids.append(random.choice(self.vectors))

    # Euclidean Distance between centroid c and vector x
    def dist(self, c, x):
        return distance.euclidean(c, x)


