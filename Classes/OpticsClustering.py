import os
from sklearn.cluster import OPTICS
from typing import Optional, Dict, Any
import numpy as np

#Avoid memory leaks
os.environ['OMP_NUM_THREADS'] = '1'

class OPTICSClusteringWrapper:
    def __init__(
        self,
        min_samples: int = 5,
        metric: str = 'euclidean',
        algorithm: str = 'auto',
        xi: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
        leaf_size: int = 30,
        n_jobs: int = -1,
    ):
        """
        Wrapper for scikit-learn's OPTICS clustering algorithm.
        Initializes the OPTICS object with given parameters.

        Args:
            min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
            metric: Distance metric to use.
            algorithm: Algorithm to compute the nearest neighbors.
            xi: Determines the steepness on the reachability plot that constitutes a cluster boundary.
            min_cluster_size: Minimum number of samples in an OPTICS cluster.
            max_eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
            leaf_size: Leaf size passed to BallTree or KDTree.
            n_jobs: Number of parallel jobs to run.
        """
        self.metric =  metric
        self.algorithm = algorithm
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.xi = xi

        self.model = OPTICS(
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            xi=xi,
            min_cluster_size=min_cluster_size,
            leaf_size=leaf_size,
            n_jobs=n_jobs
        )

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model using the given data.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            labels: Cluster labels for each data point.
        """
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

    def set_params(self, **params) -> "OPTICSClusteringWrapper":
        """
        Set the parameters of this estimator.

        Args:
            params: Dictionary of parameters.

        Returns:
            self: The updated model.
        """
        self.model.set_params(**params)
        return self