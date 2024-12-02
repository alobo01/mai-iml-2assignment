import numpy as np
from Classes.KMeans import KMeansAlgorithm
from collections import defaultdict

EPSILON = np.finfo(float).eps


def compute_log_likelihood(
        cluster_size: int,
        subcluster_size: int,
        variance: float,
        num_features: int,
        num_subclusters: int
) -> float:
    """
    Compute the log likelihood for a given cluster and subcluster.

    Args:
        cluster_size: Number of points in the cluster
        subcluster_size: Number of points in the subcluster
        variance: Variance of the cluster or subcluster
        num_features: Dimensionality of the data
        num_subclusters: Number of subclusters

    Returns:
        Log-likelihood value
    """
    if 0 <= variance <= EPSILON:
        return 0

    likelihood = (
            subcluster_size * (
            np.log(subcluster_size)
            - np.log(cluster_size)
            - 0.5 * (np.log(2 * np.pi) + num_features * np.log(variance) + 1)
    )
            + 0.5 * num_subclusters
    )
    return 0 if likelihood == np.inf else likelihood


class XMeans:
    def __init__(self, max_clusters=50, max_iterations=1000, distance_metric='euclidean', **kmeans_params):
        self.max_clusters = max_clusters
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric
        self.kmeans_params = kmeans_params
        self.centroid_cache = {}  # Cache to store centroid contributions for acceleration
        self.cluster_cache = defaultdict(list)  # Cache for cluster points

    def _initialize_kmeans(self, n_clusters: int, data: np.ndarray) -> KMeansAlgorithm:
        indices = np.random.choice(data.shape[0], n_clusters, replace=False)
        initial_centroids = data[indices]

        return KMeansAlgorithm(
            k=n_clusters,
            centroids=initial_centroids,
            distance_metric=self.distance_metric,
            max_iter=self.kmeans_params.get('max_iter', 10)
        )

    def _cache_centroid_contributions(self, cluster_labels, data):
        """
        Cache the contributions of points to centroids.
        """
        self.cluster_cache.clear()
        for i, label in enumerate(cluster_labels):
            self.cluster_cache[label].append(data[i])

    def _is_active_split(self, centroid_idx, old_centroids, new_centroids):
        """
        Determine if a centroid is active (moved significantly).
        """
        return np.linalg.norm(old_centroids[centroid_idx] - new_centroids[centroid_idx]) > EPSILON

    def determine_additional_splits(self, num_clusters, data, cluster_labels, cluster_centroids, num_features, num_subclusters, current_max_clusters):
        bic_before_split = np.zeros(num_clusters)
        bic_after_split = np.zeros(num_clusters)
        params_per_cluster = num_features + 1
        additional_clusters = 0

        for cluster_idx in range(num_clusters):
            cluster_points = np.array(self.cluster_cache[cluster_idx])
            cluster_size = cluster_points.shape[0]

            if num_clusters + additional_clusters >= current_max_clusters or cluster_size <= num_subclusters:
                continue

            cluster_variance = np.sum(
                (cluster_points - cluster_centroids[cluster_idx]) ** 2
            ) / (cluster_size - 1)

            bic_before_split[cluster_idx] = compute_log_likelihood(
                cluster_size, cluster_size, cluster_variance, num_features, 1
            ) - (params_per_cluster / 2.0) * np.log(cluster_size)

            kmeans_subclusters = self._initialize_kmeans(num_subclusters, cluster_points)
            subcluster_labels, _ = kmeans_subclusters.fit(cluster_points)
            subcluster_centroids = kmeans_subclusters.centroids

            log_likelihood = 0
            for subcluster_idx in range(num_subclusters):
                subcluster_points = cluster_points[subcluster_labels == subcluster_idx]
                subcluster_size = subcluster_points.shape[0]

                if subcluster_size <= num_subclusters:
                    continue

                subcluster_variance = np.sum(
                    (subcluster_points - subcluster_centroids[subcluster_idx]) ** 2
                ) / (subcluster_size - num_subclusters)

                log_likelihood += compute_log_likelihood(
                    cluster_size, subcluster_size, subcluster_variance,
                    num_features, num_subclusters
                )

            params_per_subcluster = num_subclusters * params_per_cluster
            bic_after_split[cluster_idx] = log_likelihood - (
                    params_per_subcluster / 2.0
            ) * np.log(cluster_size)

            if bic_before_split[cluster_idx] < bic_after_split[cluster_idx]:
                additional_clusters += 1

        return additional_clusters

    def fit(self, data: np.ndarray) -> np.ndarray:
        num_clusters = 1
        num_subclusters = 2
        num_features = data.shape[1]
        iteration = 0
        stop_splitting = False

        while not stop_splitting and iteration < self.max_iterations:
            kmeans = self._initialize_kmeans(num_clusters, data)
            cluster_labels, _ = kmeans.fit(data)
            cluster_centroids = kmeans.centroids

            self._cache_centroid_contributions(cluster_labels, data)

            additional_clusters = self.determine_additional_splits(
                num_clusters, data, cluster_labels, cluster_centroids,
                num_features, num_subclusters, self.max_clusters
            )

            additional_clusters = min(additional_clusters, self.max_clusters - num_clusters)
            num_clusters += additional_clusters

            stop_splitting = additional_clusters == 0 or num_clusters >= self.max_clusters
            iteration += 1

        final_kmeans = self._initialize_kmeans(num_clusters, data)
        self.labels_, _ = final_kmeans.fit(data)
        self.centroids = final_kmeans.centroids
        self.n_clusters = num_clusters

        return self.labels_
