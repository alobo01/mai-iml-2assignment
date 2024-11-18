import numpy as np
from Classes.KMeans import KMeansAlgorithm

class XMeansAlgorithm:
    def __init__(self, initial_k: int, max_k: int, distance_metric: str = 'euclidean', max_iter: int = 10):
        """
        X-Means clustering algorithm.

        Args:
            initial_k: Minimum number of clusters to start with.
            max_k: Maximum number of clusters.
            distance_metric: Distance metric to use ('euclidean', 'manhattan', or 'clark').
            max_iter: Maximum number of iterations for K-Means.
        """
        self.initial_k = initial_k
        self.max_k = max_k
        self.distance_metric = distance_metric
        self.max_iter = max_iter

    def fit(self, X: np.ndarray):
        """
        Fit the X-Means clustering algorithm to the data.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Final cluster labels and number of clusters.
        """
        n_samples, n_features = X.shape
        current_k = self.initial_k

        # Initialize centroids for K-Means
        centroids = X[np.random.choice(n_samples, self.initial_k, replace=False)]
        kmeans = KMeansAlgorithm(k=current_k, centroids=centroids,
                                 distance_metric=self.distance_metric, max_iter=self.max_iter)

        while True:
            # Step 2: Run K-Means
            labels, total_variance = kmeans.fit(X)
            new_centroids = []
            split = False

            # Step 3: Attempt to split each cluster
            for cluster_idx in range(current_k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) <= 1:
                    # Cannot split single-point clusters
                    new_centroids.append(kmeans.centroids[cluster_idx])
                    continue

                # Perturb centroids in opposite directions
                centroid = kmeans.centroids[cluster_idx]
                perturbation = np.random.randn(*centroid.shape) * np.std(cluster_points, axis=0)
                initial_split_centroids = np.vstack((centroid + perturbation, centroid - perturbation))

                # Step 4: Run K-Means with K=2 on the cluster
                split_kmeans = KMeansAlgorithm(k=2, centroids=initial_split_centroids,
                                               distance_metric=self.distance_metric, max_iter=self.max_iter)
                split_labels, split_variance = split_kmeans.fit(cluster_points)

                # Calculate BIC for the split and original cluster
                split_bic = self.compute_bic(cluster_points, split_variance, 2)
                original_bic = self.compute_bic(cluster_points, total_variance, 1)

                if split_bic > original_bic:
                    # Replace original centroid with the two split centroids
                    new_centroids.extend(split_kmeans.centroids)
                    split = True
                else:
                    # Retain the original centroid
                    new_centroids.append(kmeans.centroids[cluster_idx])

            # Update centroids and current_k
            new_k = len(new_centroids)
            if new_k == current_k or new_k >= self.max_k or not split:
                # Stop if convergence conditions are met
                break

            current_k = new_k
            kmeans = KMeansAlgorithm(k=current_k, centroids=np.array(new_centroids),
                                     distance_metric=self.distance_metric, max_iter=self.max_iter)

        # Final labels
        final_labels, _ = kmeans.fit(X)
        return final_labels, current_k

    @staticmethod
    def compute_bic(X: np.ndarray, variance: float, k: int) -> float:
        """
        Compute the Bayesian Information Criterion (BIC).

        Args:
            X: Input data of shape (n_samples, n_features).
            variance: Variance of the model.
            k: Number of clusters.

        Returns:
            BIC score.
        """
        n_samples, n_features = X.shape
        free_params = k * (n_features + 1)
        log_likelihood = -0.5 * n_samples * np.log(variance + np.finfo(float).eps)
        bic = log_likelihood - 0.5 * free_params * np.log(n_samples)
        return bic
