import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from Classes.XMeans import XMeans

# Assuming XMeans and KMeansAlgorithm classes are already defined and imported

def generate_data(data_type='blobs', n_samples=300, n_features=2, n_clusters=3, noise=0.1):
    """
    Generate synthetic data for benchmarking.
    :param data_type: Type of dataset ('blobs', 'moons')
    :param n_samples: Number of samples to generate
    :param n_features: Number of features
    :param n_clusters: Number of clusters
    :param noise: Noise level for 'moons' dataset
    :return: Generated data (X, y)
    """
    if data_type == 'blobs':
        return make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    elif data_type == 'moons':
        return make_moons(n_samples=n_samples, noise=noise, random_state=42)
    else:
        raise ValueError("Invalid dataset type. Choose 'blobs' or 'moons'.")


def plot_clusters(X, labels, centroids, title="Clustering Results"):
    """
    Plot the clusters and centroids.
    :param X: Input data
    :param labels: Cluster labels
    :param centroids: Cluster centroids
    :param title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.7, edgecolors='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label='Centroids')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def benchmark_xmeans(X, kmax=10, max_iter=100, distance_metric='euclidean'):
    """
    Benchmark the XMeans algorithm on a dataset.
    :param X: Input data
    :param kmax: Maximum number of clusters
    :param max_iter: Maximum number of iterations
    :param distance_metric: Distance metric to use ('euclidean', 'manhattan', etc.)
    :return: None
    """
    # Initialize XMeans with specified parameters
    xmeans = XMeans(kmax=kmax, max_iter=max_iter, distance_metric=distance_metric)

    # Fit XMeans to the data
    labels = xmeans.fit(X)

    # Plot the results
    plot_clusters(X, labels, xmeans.centroids, title="XMeans Clustering Results")


import numpy as np
import matplotlib.pyplot as plt


def generate_spherical_gaussian_clusters_no_overlap(num_clusters, num_points, dim, std_dev, min_distance):
    # Function to calculate the Euclidean distance between two points
    def distance(c1, c2):
        return np.linalg.norm(c1 - c2)

    data = []
    labels = []

    centers = []  # To store cluster centers

    for i in range(num_clusters):
        # Generate a new center for the cluster
        while True:
            new_center = np.random.rand(dim) * 10  # Scale center to be within a reasonable range
            # Check if the new center is too close to any existing center
            if all(distance(new_center, center) >= min_distance for center in centers):
                break  # The center is far enough from all others, so we accept it

        centers.append(new_center)

        # Generate points for the cluster using a spherical Gaussian distribution
        cluster_data = np.random.randn(num_points, dim) * std_dev + new_center
        data.append(cluster_data)
        labels.extend([i] * num_points)

    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels

# Example benchmarking with synthetic data
if __name__ == "__main__":
    # Example: Generate 3 clusters, each with 100 points in 2D, no overlap, with a standard deviation of 1, and minimum distance of 5 between centers
    num_clusters = 5
    num_points = 100
    dim = 2
    std_dev = 0.75
    min_distance = 4  # Minimum distance between cluster centers

    data, labels = generate_spherical_gaussian_clusters_no_overlap(num_clusters, num_points, dim, std_dev, min_distance)

    # Plot the generated dataset
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title('Spherical Gaussian Clusters (No Overlap)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # Run benchmark for XMeans
    benchmark_xmeans(data, kmax=50, max_iter=500, distance_metric='euclidean')

    # # Example with moons dataset
    # X, y = generate_data(data_type='moons', n_samples=300, n_features=2, n_clusters=2, noise=0.1)
    #
    # # Run benchmark for XMeans
    # benchmark_xmeans(X, kmax=10, max_iter=100, distance_metric='euclidean')
