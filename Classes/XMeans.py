import numpy as np
from Classes.KMeans import KMeansAlgorithm

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
    def __init__(self, max_clusters=20, max_iterations=100, distance_metric='euclidean'):
        """
        XMeans clustering algorithm.

        Args:
            max_clusters: Maximum number of clusters
            max_iterations: Maximum number of iterations for the while loop
            distance_metric: Distance metric for clustering ('euclidean', 'manhattan', 'clark')
        """
        self.max_clusters = max_clusters
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric

    def _initialize_kmeans(self, n_clusters: int, data: np.ndarray,
                           initial_centroids: np.ndarray = None) -> KMeansAlgorithm:
        """
        Initialize KMeansAlgorithm with either provided or randomly selected centroids.

        Args:
            n_clusters: Number of clusters
            data: Input data
            initial_centroids: Optional pre-defined initial centroids

        Returns:
            Initialized KMeansAlgorithm instance
        """
        # If initial centroids are provided, use them
        if initial_centroids is not None:
            return KMeansAlgorithm(
                k=n_clusters,
                centroids=initial_centroids,
                distance_metric=self.distance_metric
            )

        # If no centroids provided, randomly select from data points
        indices = np.random.choice(data.shape[0], n_clusters, replace=False)
        random_initial_centroids = data[indices]

        return KMeansAlgorithm(
            k=n_clusters,
            centroids=random_initial_centroids,
            distance_metric=self.distance_metric
        )

    def determine_additional_splits(
            self, num_clusters: int, data: np.ndarray, cluster_labels: np.ndarray,
            cluster_centroids: np.ndarray, num_features: int, num_subclusters: int
    ) -> int:
        """
        Determine whether clusters should be split into subclusters based on BIC comparison.

        Args:
            num_clusters: Number of current clusters
            data: Dataset
            cluster_labels: Labels for clusters
            cluster_centroids: Cluster centroids
            num_features: Number of features in the dataset
            num_subclusters: Number of subclusters to consider when splitting

        Returns:
            Number of new clusters to create
        """
        bic_before_split = np.zeros(num_clusters)
        bic_after_split = np.zeros(num_clusters)
        params_per_cluster = num_features + 1
        additional_clusters = 0

        for cluster_idx in range(num_clusters):
            cluster_points = data[cluster_labels == cluster_idx]
            cluster_size = cluster_points.shape[0]

            if cluster_size <= num_subclusters:
                continue

            # Compute variance and standard deviation for the current cluster
            cluster_variance = np.sum(
                (cluster_points - cluster_centroids[cluster_idx]) ** 2
            ) / (cluster_size - 1)
            cluster_std = np.sqrt(cluster_variance)

            # Compute BIC before split
            bic_before_split[cluster_idx] = compute_log_likelihood(
                cluster_size, cluster_size, cluster_variance, num_features, 1
            ) - (params_per_cluster / 2.0) * np.log(cluster_size)

            # Generate initial centroids for subclustering
            initial_centroids = np.zeros((num_subclusters, num_features))
            parent_centroid = cluster_centroids[cluster_idx]

            for subcluster_idx in range(num_subclusters):
                # Generate a random direction vector
                random_direction = np.random.randn(num_features)
                random_direction /= np.linalg.norm(random_direction)  # Normalize

                # Create two opposite centroids based on cluster standard deviation
                # Use a scaling factor to control the spread (e.g., 1.0 or 1.5 times std)
                offset = random_direction * cluster_std * (1.0 if subcluster_idx % 2 == 0 else -1.0)
                initial_centroids[subcluster_idx] = parent_centroid + offset

            # Initialize and fit KMeans for subclustering with custom initial centroids
            kmeans_subclusters = self._initialize_kmeans(num_subclusters, cluster_points, initial_centroids)
            subcluster_labels, _ = kmeans_subclusters.fit(cluster_points)
            subcluster_centroids = kmeans_subclusters.centroids

            # Compute BIC after split
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
        """
        Fit the XMeans algorithm to the data.

        Args:
            data: Input data of shape (n_samples, n_features)

        Returns:
            XMeans clustering labels
        """
        num_clusters = 2
        num_subclusters = 2
        num_features = data.shape[1]
        iteration = 0

        while iteration < self.max_iterations:
            # Initialize and fit KMeans
            kmeans = self._initialize_kmeans(num_clusters, data)
            cluster_labels, _ = kmeans.fit(data)
            cluster_centroids = kmeans.centroids

            # Check for additional cluster splits
            additional_clusters = self.determine_additional_splits(
                num_clusters, data, cluster_labels, cluster_centroids,
                num_features, num_subclusters
            )

            new_number_of_clusters = num_clusters + additional_clusters
            if additional_clusters == 0 or new_number_of_clusters > self.max_clusters:
                break
            elif new_number_of_clusters == self.max_clusters:
                num_clusters = new_number_of_clusters
                break
            num_clusters = new_number_of_clusters

            iteration += 1

        # Perform final clustering with determined number of clusters
        final_kmeans = self._initialize_kmeans(num_clusters, data)
        self.labels_, _ = final_kmeans.fit(data)
        self.centroids = final_kmeans.centroids
        self.n_clusters = num_clusters

        return self.labels_
