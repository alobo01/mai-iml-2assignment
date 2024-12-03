from typing import Union

import numpy as np

class FuzzyCMeans:
    def __init__(self, n_clusters: int=3, fuzziness: float=2, max_iter: int=150, error: float=1e-5, random_state:Union[int,None]=None, omega=0.95, suppression_factor=0.2, rho=0.6):
        """
        Initialize the Fuzzy C-means clustering model.

        Parameters:
        - n_clusters: int, default=3
            Number of clusters.
        - m: float, default=2
            Fuzziness parameter (>1).
        - max_iter: int, default=150
            Maximum number of iterations.
        - error: float, default=1e-5
            Stopping criterion threshold.
        - random_state: int or None, default=None
            Random seed for reproducibility.
        - omega: float, default=0.95
            Suppression factor for multimodality.
        - suppression_factor: float, default=0.2
            Default suppression factor for non-generalized case.
        - rho: float, default=0.6
            Linear decay factor for generalized s-FCM.
        """
        self.n_clusters = n_clusters
        self.m = fuzziness
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        # FCM with improved partition
        # Omega between 0.9 and 0.99
        self.omega = omega
        # supressed FCM
        self.suppression_factor = suppression_factor
        self.rho = rho  # Linear decay factor for f(u_w)


        self.centers = None       # Cluster centers
        self.U = None             # Membership matrix
        self.n_samples = None     # Number of data samples
        self.n_features = None    # Number of features
        # FCM with improved partition
        self.a = None # Parameter to supress multimodality at borders.
        # supressed FCM
        if isinstance(suppression_factor,float):
            self.alpha = suppression_factor
        else:
            # Generalized s-FCM
            self.alpha = None


    def fit(self, X):
        """
        Compute fuzzy c-means clustering.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Training instances to cluster.
        """
        X = np.array(X)
        self.n_samples, self.n_features = X.shape

        # Initialize membership matrix U with random values
        if self.random_state is not None:
            np.random.seed(self.random_state)
        U = np.random.dirichlet(np.ones(self.n_clusters), size=self.n_samples)
        mu = U
        for iteration in range(self.max_iter):
            U_old = U.copy()
            # Compute cluster centers
            #um = U ** self.m
            #centers = (um.T @ X) / np.atleast_2d(um.sum(axis=0)).T
            # Supressed FCM
            mum = mu ** self.m
            centers = (mum.T @ X) / np.atleast_2d(mum.sum(axis=0)).T
            # Compute distance matrix
            dist = np.zeros((self.n_samples, self.n_clusters))
            for i in range(self.n_clusters):
                dist[:, i] = np.linalg.norm(X - centers[i], axis=1)
            dist = np.fmax(dist, np.finfo(np.float64).eps)
            a = self._calculate_alpha(dist)

            # Update membership matrix U

            #U = self._calculate_original_u(dist)
            U = self._calculate_improved_partition_u(dist,a)
            # Check convergence
            if np.linalg.norm(U - U_old) < self.error:
                break

            # Supressed FCM
            mu = self._calculate_mu(U)


        self.centers = centers
        self.U = U
        labels = np.argmax(U, axis=1)
        return labels

    def _calculate_mu(self, U):
        """
        Calculate mu based on U and dynamically computed alpha_k.

        Parameters:
        - U: Membership matrix

        Returns:
        - Updated mu matrix.
        """
        # Find the index of the maximum membership value for each sample
        max_indices = np.argmax(U, axis=1)
        rows = np.arange(U.shape[0])

        # Calculate u_w (winner membership) for each sample
        u_w = U[rows, max_indices]

        # Compute alpha_k for each sample based on the provided formula
        alpha_k = (1 - u_w + self.rho**(2 / (1 - self.m)) * u_w**((3 - self.m) / (1 - self.m)))**(-1)

        # Initialize mu with the formula for "non-winners": mu = alpha_k * u_ik
        mu = alpha_k[:, np.newaxis] * U

        # Update mu for the winners: mu = 1 - alpha_k + alpha_k * u_ik
        mu[rows, max_indices] = 1 - alpha_k + alpha_k * U[rows, max_indices]

        return mu

    def _calculate_original_u(self, dist):
        """
        Calculate the original membership matrix U based on the original equation.

        Parameters:
        - dist: np.ndarray (c x n)
          Matrix of distances where dist[i, k] is the distance between cluster i and data point k.
        - m: float
          Fuzziness parameter (m > 1).

        Returns:
        - U: np.ndarray (c x n)
          Membership matrix.
        """
        exponent = 2 / (self.m - 1)
        temp = dist[:, :, np.newaxis] / dist[:, np.newaxis, :]  # Compute distance ratios
        temp = temp ** exponent  # Apply exponent
        U = 1 / temp.sum(axis=2)  # Normalize across clusters
        return U

    def _calculate_improved_partition_u(self, dist, a):
        """
        Calculate the improved membership matrix U based on the modified equation.

        Parameters:
        - dist: np.ndarray (c x n)
          Matrix of distances where dist[i, k] is the distance between cluster i and data point k.
        - a: np.ndarray (n,)
          Penalty or offset term for each data point (1D array of length n).
        - m: float
          Fuzziness parameter (m > 1).

        Returns:
        - U: np.ndarray (c x n)
          Membership matrix.
        """
        exponent = -1 / (self.m - 1)
        dist_squared = dist ** 2  # Square the distances
        temp = dist_squared - a[:, np.newaxis]  # Subtract the penalty term for each point

        # Prevent negative or zero values to avoid invalid computations
        temp[temp <= 0] = np.finfo(float).eps

        temp = temp ** exponent  # Apply exponent
        U = temp / temp.sum(axis=1)[:, np.newaxis]  # Normalize across clusters
        return U

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            New data to predict.

        Returns:
        - labels: array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        U = self._calculate_u(X)
        self.U = U
        labels = np.argmax(U, axis=1)
        return labels

    def _calculate_alpha(self, dist):
        """
        Compute the penalty term a for each data point based on the equation:
        a_k = omega * min(d_ik^2, i = 1...c)

        Parameters:
        - dist: np.ndarray (n_samples x n_clusters)
          Matrix of distances where dist[k, i] is the distance between data point k and cluster center i.

        Returns:
        - a: np.ndarray (n_samples,)
          Penalty term for each data point.
        """
        dist_squared = dist ** 2  # Square the distances
        min_dist_squared = np.min(dist_squared, axis=1)  # Find the minimum squared distance for each point
        a = self.omega * min_dist_squared  # Multiply by omega
        return a

    def _learning_rate(self):
        pass
    def _calculate_u(self, X):
        """
        Calculate the membership matrix U for new data X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Data to calculate membership for.

        Returns:
        - U: array, shape (n_samples, n_clusters)
            Membership matrix.
        """
        X = np.array(X)
        dist = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            dist[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        a = self._calculate_alpha(dist)

        U = self._calculate_improved_partition_u(dist,a)
        return U


if __name__ == "__main__":
    # Create synthetic data
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Instantiate the model
    fcm = FuzzyCMeans(n_clusters=4, m=2, max_iter=100, error=1e-5, random_state=42)

    # Fit the model
    fcm.fit(X)
    # Predict cluster labels
    labels = fcm.predict(X)

    # Plotting (optional)
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], marker='*', s=200, c='red')
    plt.show()

    # Create synthetic 1D data: Combination of 4 Gaussian distributions
    np.random.seed(42)
    data1 = np.random.normal(loc=-3, scale=0.5, size=20)
    data2 = np.random.normal(loc=0, scale=0.5, size=20)
    data3 = np.random.normal(loc=3, scale=0.5, size=20)

    X_1D = np.concatenate([data1, data2, data3]).reshape(-1, 1)

    # Create labels for coloring
    labels = np.concatenate([np.zeros(len(data1)), np.ones(len(data2)), np.full(len(data3), 2)])

    # Plot the data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_1D, np.zeros_like(X_1D), c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.yticks([])
    plt.title("1D Data Distribution with Color Labels")
    plt.xlabel("X_1D Values")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Group 1", "Group 2", "Group 3"], title="Groups")
    plt.show()

    # Define different fuzziness levels (m-values)
    fuzziness_levels = [1.5, 2.0, 3.0, 5.0]

    # Plot membership functions for 1D data at different fuzziness levels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, m in enumerate(fuzziness_levels):
        fcm = FuzzyCMeans(n_clusters=3, m=m, max_iter=5000, error=1e-5, random_state=42)
        fcm.fit(X_1D)
        U = fcm.U  # Membership matrix

        ax = axes[idx // 2, idx % 2]
        for i in range(U.shape[1]):  # For each cluster
            ax.plot(np.arange(U.shape[0]), U[:, i], label=f'Cluster {i + 1}', linestyle='--')

        ax.set_title(f'm = {m}')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Membership Probability')
        ax.legend()

    plt.tight_layout()
    plt.show()