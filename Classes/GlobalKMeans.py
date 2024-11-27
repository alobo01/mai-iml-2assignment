from typing import Callable
from functools import cache
import numpy as np

class GlobalKMeansAlgorithm:
    def __init__(self, k: int, distance_metric: str = 'euclidean', max_iter: int = 10):
        self.k = k
        self.distance_metric = distance_metric
        self.distance = self.get_distance(distance_metric)
        self.max_iter = max_iter
        self.centroids = None

    def get_distance(self, distance_metric) -> Callable[[np.ndarray, np.ndarray],np.ndarray]:
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

    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the Global K-means algorithm.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Initialized centroids of shape (k, n_features)
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize first centroid as the mean of all points
        best_centroids = np.mean(X, axis=0, keepdims=True)

        # Step 2: Iteratively add centroids
        for k_prime in range(2, self.k + 1):
            best_variance = float('inf')
            candidate_centroids = None

            # Try each point as the new centroid
            for i in range(n_samples):
                # Current centroids: previous best centroids plus current candidate
                current_centroids = np.vstack([best_centroids, X[i]])

                # Run k'-means with these initial centroids
                temp_centroids = current_centroids.copy()

                # Run k'-means to convergence (using a reasonable max_iter)
                for _ in range(self.max_iter):
                    # Assign points to nearest centroids
                    distances = self.distance(X, temp_centroids)
                    labels = np.argmin(distances, axis=1)

                    # Update centroids
                    new_temp_centroids = np.zeros((k_prime, n_features))
                    for j in range(k_prime):
                        cluster_points = X[labels == j]
                        if len(cluster_points) > 0:
                            new_temp_centroids[j] = cluster_points.mean(axis=0)
                        else:
                            new_temp_centroids[j] = temp_centroids[j]

                    # Check for convergence
                    if np.allclose(temp_centroids, new_temp_centroids):
                        break

                    temp_centroids = new_temp_centroids

                # Compute total variance for this configuration
                current_variance = self.compute_total_variance(X, labels, k_prime, temp_centroids)

                # Update best solution if current variance is lower
                if current_variance < best_variance:
                    best_variance = current_variance
                    candidate_centroids = temp_centroids

            # Update best centroids for k_prime clusters
            best_centroids = candidate_centroids

        return best_centroids

    def compute_total_variance(self, X: np.ndarray, labels: np.ndarray, k: int = None, centroids: np.ndarray = None) -> float:
        """
        Compute the total within-cluster variance (E).

        Args:
            X: Input data of shape (n_samples, n_features)
            labels: Cluster labels for each data point

        Returns:
            Total within-cluster variance (E)
        """
        k = self.k if not k else k
        centroids = self.centroids if centroids is None else centroids

        E = 0.0
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                # Calculate squared distances to centroid for all points in cluster
                centroid = centroids[j]
                # Reshape centroid to (1, n_features) for broadcasting
                squared_distances = np.sum(self.distance(cluster_points, centroid[np.newaxis, :]) ** 2, axis=1)
                E += np.sum(squared_distances)
        return E

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Fit K-means clustering.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Labels for each data point
        """
        # Initialize centroids using K-means++
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign points to nearest centroids
            distances = self.distance(X, self.centroids)
            labels = np.argmin(distances, axis=1)

        # Compute total within-cluster variance
        E = self.compute_total_variance(X, labels)

        return labels, E