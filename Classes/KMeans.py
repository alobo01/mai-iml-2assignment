import numpy as np
import pandas as pd

class KMeansAlgorithm:
    def __init__(self, k, centroids, distance_metric):
        self.k = k
        self.centroids = centroids
        self.distance_metric = distance_metric
        self.distance = self.get_distance(distance_metric)

    def get_distance(self, distance_metric):
        if distance_metric == 'euclidean':
            return self.euclidean_distance
        elif distance_metric == 'manhattan':
            return self.manhattan_distance
        elif distance_metric == 'clark':
            return self.clark_distance
        else:
            raise ValueError('Invalid distance metric specified.')

    def euclidean_distance(self, X, centroids):
        return np.sqrt(np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=-1))

    def manhattan_distance(self, X, centroids):
        return np.sum(np.abs(X[:, None, :] - centroids[None, :, :]), axis=-1)

    def clark_distance(self, X, centroids):
        return np.sqrt(np.sum(((X[:, None, :] - centroids[None, :, :]) / (X[:, None, :] + centroids[None, :, :]))**2, axis=-1))

    def fit(self, X):
        distances = self.distance(X, self.centroids)
        labels = np.argmin(distances, axis=1)

        for j in range(self.k):
            cluster_points = X[labels == j]
            self.centroids[j] = cluster_points.mean(axis=0)

        X['cluster'] = labels
        return X