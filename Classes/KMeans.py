from typing import Callable

import numpy as np

class KMeansAlgorithm:
    def __init__(self, k, centroids, distance_metric, max_iter = 1):
        self.k = k
        self.centroids = centroids
        self.distance_metric = distance_metric
        self.distance = self.get_distance(distance_metric)
        self.max_iter = max_iter

    def get_distance(self, distance_metric) -> Callable[[np.ndarray[float], np.ndarray[float]],np.ndarray[float]]:
        if distance_metric == 'euclidean':
            return self.euclidean_distance
        elif distance_metric == 'manhattan':
            return self.manhattan_distance
        elif distance_metric == 'clark':
            return self.clark_distance
        else:
            raise ValueError('Invalid distance metric specified.')

    @staticmethod
    def euclidean_distance(X, centroids):
        return np.sqrt(np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=-1))

    @staticmethod
    def manhattan_distance(X, centroids):
        return np.sum(np.abs(X[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=-1)

    @staticmethod
    def clark_distance(X, centroids):
        denominator = X[:, np.newaxis, :] + centroids[np.newaxis, :, :]
        denominator[denominator == 0] = np.finfo(float).eps  # Avoid division by zero
        distances = np.sqrt(np.sum(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) / denominator) ** 2, axis=-1))
        return distances

    def fit(self, X):
        it = 0
        while it < self.max_iter:
            distances = self.distance(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            for j in range(self.k):
                cluster_points = X[labels == j]
                self.centroids[j] = cluster_points.mean(axis=0)

            it = it + 1

        return labels