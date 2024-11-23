from typing import Callable, Tuple, Any
import numpy as np


class KMeansPPAlgorithm:
    def __init__(self, k: int, distance_metric: str = 'euclidean', max_iter: int = 10):
        self.k = k
        self.distance_metric = distance_metric
        self.distance = self.get_distance(distance_metric)
        self.max_iter = max_iter
        self.centroids = None

    def get_distance(self, distance_metric) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
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
        return np.sqrt(np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=-1))

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
        Initialize centroids using the K-means++ algorithm.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Initialized centroids of shape (k, n_features)
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))

        # Step 1a: Choose first centroid randomly
        first_centroid_idx = np.random.randint(n_samples)
        centroids[0] = X[first_centroid_idx]

        # Steps 1b and 1c: Choose remaining centroids
        for i in range(1, self.k):
            # Calculate distances to nearest centroids for all points
            distances = self.distance(X, centroids[:i])
            min_distances = np.min(distances, axis=1)

            # Calculate probabilities for each point
            probabilities = min_distances ** 2
            probabilities = probabilities / np.sum(probabilities)

            # Choose next centroid based on calculated probabilities
            next_centroid_idx = np.random.choice(n_samples, p=probabilities)
            centroids[i] = X[next_centroid_idx]

        return centroids

    def compute_total_variance(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the total within-cluster variance (E).

        Args:
            X: Input data of shape (n_samples, n_features)
            labels: Cluster labels for each data point

        Returns:
            Total within-cluster variance (E)
        """
        E = 0.0
        for j in range(self.k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                # Calculate squared distances to centroid for all points in cluster
                centroid = self.centroids[j]
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
        n_samples, n_features = X.shape

        # Initialize centroids using K-means++
        self.centroids = self.initialize_centroids(X)

        # Standard K-means algorithm
        for _ in range(self.max_iter):
            # Assign points to nearest centroids
            distances = self.distance(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros((self.k, n_features))
            for j in range(self.k):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # Compute total within-cluster variance
        E = self.compute_total_variance(X, labels)

        return labels, E