import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, davies_bouldin_score, cluster

class EvaluationUtils:
    @staticmethod
    def purity_score(y_true, y_pred):
        """
        Compute purity score for clustering results.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted cluster labels.

        Returns:
            Purity score (float).
        """
        # Compute contingency matrix (confusion matrix)
        contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
        # Return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

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
            - Purity score
        """
        # Compute metrics
        ari = adjusted_rand_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        dbi = davies_bouldin_score(X, y_pred)
        purity = EvaluationUtils.purity_score(y_true, y_pred)

        # Return metrics as a dictionary
        return {
            'ARI': ari,
            'F1 Score': f1,
            'DBI': dbi,
            'Purity': purity
        }
