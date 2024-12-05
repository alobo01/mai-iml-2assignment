from math import log
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from Classes.KMeans import KMeansAlgorithm

class XMeans:
    """
    An implementation of the X-Means clustering algorithm with KMeans++ initialization.

    Parameters:
    - max_clusters: Maximum number of clusters.
    - tolerance: Tolerance for convergence.
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'clark').
    - repeat_kmeans: Number of times to repeat KMeans for robust results.
    """

    def __init__(self, max_clusters=20, tolerance=0.001, repeat_kmeans=1):
        self.max_clusters = max_clusters
        self.tolerance = tolerance
        self.repeat_kmeans = repeat_kmeans

        self.clusters = []
        self.centroids = None
        self.data = None

    def _initialize_kmeans_plus_plus(self, data, num_clusters):
        """
        Initializes cluster centroids using KMeans++.
        """
        kmeans = KMeansAlgorithm(
            k=num_clusters, centroids=None, max_iter=100
        )
        kmeans.fit(data)
        return kmeans.centroids

    def fit(self, data):
        """
        Performs the X-Means clustering algorithm on the provided dataset.
        """
        self.data = np.array(data)
        if self.centroids is None:
            self.centroids = self._initialize_kmeans_plus_plus(data, 2)

        while len(self.centroids) <= self.max_clusters:
            current_cluster_count = len(self.centroids)
            self.clusters, self.centroids = self._optimize_parameters(self.centroids)
            updated_centroids = self._optimize_structure(self.clusters, self.centroids)
            if current_cluster_count == len(updated_centroids):
                break
            else:
                self.centroids = updated_centroids

        self.clusters, self.centroids = self._optimize_parameters(self.centroids)
        return self._generate_labels()

    def _generate_labels(self):
        """
        Generates cluster labels for the dataset based on final clusters.
        """
        labels = np.zeros(len(self.data), dtype=int)
        for cluster_index, cluster_points in enumerate(self.clusters):
            for point_index in cluster_points:
                labels[point_index] = cluster_index
        return labels

    def _find_optimal_parameters(self, subset_data):
        """
        Finds optimal parameters for splitting a cluster into two sub-clusters.
        """
        subset_data = np.array(subset_data)
        best_wce = float('inf')
        best_centroids, best_clusters = None, None

        for _ in range(self.repeat_kmeans):
            initial_centroids = self._initialize_kmeans_plus_plus(subset_data, 2)
            kmeans = KMeansAlgorithm(
                k=2, centroids=initial_centroids, max_iter=100
            )
            labels = kmeans.fit(subset_data)
            wce = kmeans.compute_total_variance(subset_data, labels)

            if wce < best_wce:
                best_wce = wce
                best_centroids = kmeans.centroids
                best_clusters = self._assign_clusters(subset_data, best_centroids)

        return best_clusters, best_centroids, best_wce

    def _assign_clusters(self, data, centroids):
        distances = euclidean_distances(data, centroids)
        return [
            np.where(distances[:, i] == np.min(distances, axis=1))[0].tolist()
            for i in range(len(centroids))
        ]

    def _optimize_parameters(self, centroids, indices=None):
        """
        Refines parameters of the current centroids to improve clustering.
        """
        if indices and len(indices) == 1:
            return [[indices[0]]], self.data[indices[0]], 0.0

        subset_data = self.data if indices is None else self.data[indices]
        if centroids is None:
            clusters, centroids, wce = self._find_optimal_parameters(subset_data)
        else:
            kmeans = KMeansAlgorithm(
                k=len(centroids), centroids=np.array(centroids), max_iter=100
            )
            centroids = kmeans.centroids
            clusters = self._assign_clusters(subset_data, centroids)

        if indices:
            clusters = self._map_local_to_global_clusters(clusters, indices)

        return clusters, centroids

    def _map_local_to_global_clusters(self, local_clusters, indices):
        """
        Maps local cluster indices back to the original dataset indices.
        """
        global_clusters = []
        for cluster in local_clusters:
            global_clusters.append([indices[idx] for idx in cluster])
        return global_clusters

    def _optimize_structure(self, clusters, centroids):
        """
        Evaluates the structure of the clusters to determine splits or merges.
        """
        new_centroids = []
        remaining_centroids = self.max_clusters - len(centroids)

        for i, cluster in enumerate(clusters):
            child_clusters, child_centroids = self._optimize_parameters(None, cluster)
            if len(child_clusters) > 1:
                parent_bic = self._compute_bic([cluster], [centroids[i]])
                child_bic = self._compute_bic(child_clusters, child_centroids)

                if parent_bic < child_bic and remaining_centroids > 0:
                    new_centroids.extend(child_centroids)
                    remaining_centroids -= 1
                else:
                    new_centroids.append(centroids[i])
            else:
                new_centroids.append(centroids[i])

        return new_centroids

    def _compute_bic(self, clusters, centroids):
        """
        Computes the Bayesian Information Criterion (BIC) for a given clustering.
        """
        scores = []
        dimensions = len(self.data[0])
        total_points = 0
        sigma_squared = 0.0
        num_clusters = len(clusters)

        for i, cluster in enumerate(clusters):
            cluster_size = len(cluster)
            total_points += cluster_size
            for point_index in cluster:
                sigma_squared += np.sum((self.data[point_index] - centroids[i]) ** 2)

        if total_points - num_clusters > 0:
            sigma_squared /= (total_points - num_clusters)
            p = (num_clusters - 1) + dimensions * num_clusters + 1
            log_likelihood = 0.0

            for i, cluster in enumerate(clusters):
                cluster_size = len(cluster)
                if sigma_squared > 0:
                    log_likelihood = (
                            cluster_size * log(cluster_size)
                            - cluster_size * log(total_points)
                            - cluster_size * 0.5 * log(2 * np.pi)
                            - cluster_size * dimensions * 0.5 * log(sigma_squared)
                            - (cluster_size - num_clusters) * 0.5
                    )
                scores.append(log_likelihood - 0.5 * p * log(total_points))

        return sum(scores)

