import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score
)


class EvaluationUtils:
    @staticmethod
    def evaluate(X, y_true, y_pred):
        """
        Compute evaluation metrics for clustering.

        Args:
            X: Data of shape (n_samples, n_features).
            y_true: Ground truth labels.
            y_pred: Predicted cluster labels.

        Returns:
            A dictionary containing:
            - Adjusted Rand Index (ARI)
            - F1 Score (macro)
            - Davies-Bouldin Index (DBI)
            - Silhouette Score
            - Calinski-Harabasz Score
        """
        # Compute metrics
        ari = adjusted_rand_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        dbi = davies_bouldin_score(X, y_pred)
        silhouette = silhouette_score(X, y_pred)
        calinski = calinski_harabasz_score(X, y_pred)

        # Return metrics as a dictionary
        return {
            'ARI': ari,
            'F1 Score': f1,
            'DBI': dbi,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': dbi  # Adding this explicitly to match the column name
        }
