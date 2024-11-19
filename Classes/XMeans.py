import numpy as np
from Classes.KMeans import KMeansAlgorithm

EPS = np.finfo(float).eps

def loglikelihood(R, R_n, variance, M, K):
    """
    Compute the log likelihood for a given cluster and subcluster.
    :param R: (int) size of cluster
    :param R_n: (int) size of subcluster
    :param variance: (float) variance estimate
    :param M: (float) number of features (dimensionality of the data)
    :param K: (float) number of clusters
    :return: (float) loglikelihood value
    """
    if 0 <= variance <= EPS:
        return 0
    res = R_n * (np.log(R_n) - np.log(R) - 0.5 * (np.log(2 * np.pi) + M * np.log(variance) + 1)) + 0.5 * K
    return 0 if res == np.inf else res


def get_additional_k_split(K, X, clst_labels, clst_centers, n_features, K_sub, k_means_args):
    """
    Determine whether a cluster should be split into subclusters based on BIC comparison.
    :param K: (int) number of clusters
    :param X: (np.ndarray) dataset
    :param clst_labels: (np.ndarray) labels for clusters
    :param clst_centers: (np.ndarray) cluster centroids
    :param n_features: (int) number of features in the dataset
    :param K_sub: (int) number of subclusters to attempt splitting
    :param k_means_args: (dict) arguments for KMeans
    :return: (int) number of new clusters to create
    """
    bic_before_split = np.zeros(K)
    bic_after_split = np.zeros(K)
    clst_n_params = n_features + 1
    add_k = 0

    for clst_index in range(K):
        clst_points = X[clst_labels == clst_index]
        clst_size = clst_points.shape[0]
        if clst_size <= K_sub:
            continue

        clst_variance = np.sum((clst_points - clst_centers[clst_index]) ** 2) / float(clst_size - 1)
        bic_before_split[clst_index] = loglikelihood(clst_size, clst_size, clst_variance, n_features, 1) - clst_n_params / 2.0 * np.log(clst_size)

        # Run KMeans on the cluster to split it further
        kmeans_subclst = KMeansAlgorithm(K_sub, np.random.randn(K_sub, n_features), **k_means_args)
        subclst_labels, _ = kmeans_subclst.fit(clst_points)
        subclst_centers = kmeans_subclst.centroids

        log_likelihood = 0
        for subclst_index in range(K_sub):
            subclst_points = clst_points[subclst_labels == subclst_index]
            subclst_size = subclst_points.shape[0]
            if subclst_size <= K_sub:
                continue
            subclst_variance = np.sum((subclst_points - subclst_centers[subclst_index]) ** 2) / float(subclst_size - K_sub)
            log_likelihood += loglikelihood(clst_size, subclst_size, subclst_variance, n_features, K_sub)

        subclst_n_params = K_sub * clst_n_params
        bic_after_split[clst_index] = log_likelihood - subclst_n_params / 2.0 * np.log(clst_size)

        if bic_before_split[clst_index] < bic_after_split[clst_index]:
            add_k += 1

    return add_k


class XMeans:
    def __init__(self, kmax=50, max_iter=1000, distance_metric='euclidean', **k_means_args):
        """
        :param kmax: maximum number of clusters that XMeans can divide the data into
        :param max_iter: maximum number of iterations for the `while` loop (hard limit)
        :param distance_metric: Distance metric used for clustering ('euclidean', 'manhattan', etc.)
        :param k_means_args: Additional parameters for KMeansAlgorithm
        """
        self.KMax = kmax
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.k_means_args = k_means_args

    def fit(self, X: np.ndarray):
        """
        Fit the XMeans algorithm to the data.
        :param X: Input data of shape (n_samples, n_features)
        :return: XMeans clustering labels
        """
        K = 1
        K_sub = 2
        K_old = K
        n_features = X.shape[1]
        stop_splitting = False
        iter_num = 0

        while not stop_splitting and iter_num < self.max_iter:
            K_old = K
            kmeans = KMeansAlgorithm(K, np.random.randn(K, n_features), distance_metric=self.distance_metric, **self.k_means_args)
            clst_labels, _ = kmeans.fit(X)
            clst_centers = kmeans.centroids

            # Check if additional splits are needed
            add_k = get_additional_k_split(K, X, clst_labels, clst_centers, n_features, K_sub, self.k_means_args)
            K += add_k

            # Stop if no new clusters are added or maximum clusters reached
            stop_splitting = K_old == K or K >= self.KMax
            iter_num += 1

        # Final clustering with determined number of clusters
        kmeans_final = KMeansAlgorithm(K_old, np.random.randn(K_old, n_features), distance_metric=self.distance_metric, **self.k_means_args)
        self.labels_, _ = kmeans_final.fit(X)
        self.centroids = kmeans_final.centroids
        self.n_clusters = K_old
        return self.labels_
