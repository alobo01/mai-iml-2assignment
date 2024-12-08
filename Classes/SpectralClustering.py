import os

from sklearn.cluster import SpectralClustering
from typing import Optional, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler


# X_normalized is now the normalized dataset


#Avoid memory leaks
os.environ['OMP_NUM_THREADS'] = '1'

class SpectralClusteringWrapper:
    def __init__(
        self,
        n_clusters: int = 8,
        eigen_solver: str = 'auto',
        random_state: Optional[int] = None,
        n_init: int = 1,
        affinity: str = 'nearest_neighbors',
        n_neighbors: int = 10,
        eigen_tol: float = 0.0,
        assign_labels: str = 'kmeans',
        n_jobs: int = -1
    ):
        """
        Wrapper for scikit-learn's SpectralClustering class.
        Initializes the SpectralClustering object with given parameters.
        """

        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.eigen_solver = eigen_solver
        self.assign_labels = assign_labels

        self.model = SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            n_init=n_init,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray) -> "SpectralClusteringWrapper":
        """
        Fit the model using the given data.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            labels: Cluster labels for each data point.
        """
        # Assuming X is your dataset
        #scaler = StandardScaler()
        #X_normalized = scaler.fit_transform(X)
        return self.model.fit_predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep: Whether to include nested parameters.

        Returns:
            Dictionary of parameters.
        """
        return self.model.get_params(deep)

    def set_params(self, **params) -> "SpectralClusteringWrapper":
        """
        Set the parameters of this estimator.

        Args:
            params: Dictionary of parameters.

        Returns:
            self: The updated model.
        """
        self.model.set_params(**params)
        return self
