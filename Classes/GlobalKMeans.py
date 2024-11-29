from typing import Callable, List, Tuple
from functools import cache
import numpy as np
from sklearn.decomposition import PCA

class GlobalKMeansAlgorithm:
    def __init__(self, k: int, distance_metric: str = 'euclidean', max_iter: int = 10, n_buckets: int = None):
        self.k = k
        self.distance_metric = distance_metric
        self.distance = self.get_distance(distance_metric)
        self.max_iter = max_iter
        self.centroids = None
        self.n_buckets = n_buckets if n_buckets else k*2

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

    def initialize_candidate_points(self, X: np.ndarray) -> np.ndarray:
        """
        Construct a k-d tree and select candidate points using PCA-based partitioning.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Candidate points for centroid initialization
        """

        def recursive_partition(buckets: List[np.ndarray]):
            """
            Recursively partition data using PCA and create bucket centroids.

            Args:
                buckets: List of the current partitions of the data (buckets)
            """
            # Select the bucket with most samples as the one to partition
            data = buckets[0]
            data_index = 0
            for i in range(1, len(buckets)):
                bucket = buckets[i]
                if bucket.shape[0] > data.shape[0]:
                    data = bucket
                    data_index = i

            # Compute principal component direction and project the data
            pca = PCA(n_components=1)
            pca.fit(data)
            projections = pca.transform(data)

            # Split data based on projection
            # The mean of the projection is always 0, so we split based on which side of the mean the projection of each point is
            left_mask = [projection[0] <= 0 for projection in projections]
            right_mask = [~mask for mask in left_mask]

            left_data = data[left_mask, :]
            right_data = data[right_mask, :]

            # Remove data from buckets and add partitions
            buckets.pop(data_index)
            buckets.append(left_data)
            buckets.append(right_data)


        buckets = [X]

        # Generate buckets using k-d tree partitioning
        for num_buckets in range (self.n_buckets):
            recursive_partition(buckets)

        candidate_points = [np.mean(bucket, axis=0) for bucket in buckets]

        return np.array(candidate_points)

    def calculate_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the Global K-means algorithm with k-d tree candidate points.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Initialized centroids of shape (k, n_features)
        """
        n_samples, n_features = X.shape

        # Get candidate points from k-d tree initialization
        candidate_points = self.initialize_candidate_points(X)
        n_candidate_points = candidate_points.shape[0]

        # Pre-compute pair-wise squared distances
        squared_distances = self.distance(candidate_points, candidate_points) ** 2

        # Step 1: Initialize first centroid as the mean of all points
        best_centroids = np.mean(X, axis=0, keepdims=True)
        candidate_labels = np.zeros(n_candidate_points, dtype=int)
        distance_to_centroids = self.distance(candidate_points, best_centroids) ** 2

        # Step 2: Iteratively add centroids
        for k_prime in range(2, self.k + 1):
            # Compute the guaranteed error reduction for each candidate point
            error_reductions = []
            for n in range(n_candidate_points):
                reductions = [max(distance_to_centroids[i, candidate_labels[i]] - squared_distances[n, i], 0) for i in range(n_candidate_points)]
                error_reduction = sum(reductions)
                error_reductions.append(error_reduction)

            best_candidate_index = np.argmax(error_reductions)
            best_candidate = candidate_points[best_candidate_index]
            # Current centroids: previous best centroids plus best candidate
            current_centroids = np.vstack([best_centroids, best_candidate])

            # Run k'-means to convergence
            for _ in range(self.max_iter):
                # Assign points to nearest centroids
                distances = self.distance(X, current_centroids)
                labels = np.argmin(distances, axis=1)

                # Update centroids
                temp_centroids = np.zeros((k_prime, n_features))
                for j in range(k_prime):
                    cluster_points = X[labels == j]
                    if len(cluster_points) > 0:
                        temp_centroids[j] = cluster_points.mean(axis=0)
                    else:
                        temp_centroids[j] = current_centroids[j]

                # Check for convergence
                if np.allclose(current_centroids, temp_centroids):
                    break

                current_centroids = temp_centroids

            # Update best centroids for k_prime clusters
            best_centroids = current_centroids
            distance_to_centroids = self.distance(candidate_points, best_centroids)
            candidate_labels = np.argmin(distance_to_centroids, axis=1)
            distance_to_centroids = distance_to_centroids ** 2

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

    # Renamed the previous fit method to use the new calculate_centroids
    def fit(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Fit K-means clustering.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Labels for each data point and total variance
        """
        # Initialize centroids using the new method
        self.centroids = self.calculate_centroids(X)

        # Assign points to nearest centroids
        distances = self.distance(X, self.centroids)
        labels = np.argmin(distances, axis=1)

        # Compute total within-cluster variance
        E = self.compute_total_variance(X, labels)

        return labels, E