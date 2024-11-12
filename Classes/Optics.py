import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from itertools import product

from Classes.Reader import DataPreprocessor


class OpticsClusterAnalyzer:
    def __init__(self, df, label_column=None):
        """
        Initialize the OPTICS clustering analyzer.

        Parameters:
        df (pandas.DataFrame): Input dataframe
        label_column (str): Name of the column containing true labels (if any)
        """
        self.df = df.copy()
        self.label_column = label_column
        self.X = None
        self.true_labels = None
        self.best_model = None
        self.best_params = None
        self.best_score = float('-inf')
        self.results = []

        self._prepare_data()

    def _prepare_data(self):
        """Prepare data by separating features and labels, and scaling features."""
        if self.label_column and self.label_column in self.df.columns:
            self.true_labels = self.df[self.label_column]
            feature_cols = [col for col in self.df.columns if col != self.label_column]
            self.X = self.df[feature_cols]
        else:
            self.X = self.df

        # Scale the features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def grid_search(self, param_grid=None):
        """
        Perform grid search over specified parameters.

        Parameters:
        param_grid (dict): Dictionary of parameters to try. If None, uses default grid.
        """
        if param_grid is None:
            param_grid = {
                'min_samples': [5, 10, 20],
                'max_eps': [np.inf, 5.0, 2.0],
                'min_cluster_size': [5, 10, 20],
                'xi': [0.05, 0.1, 0.2]
            }

        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v))
                              for v in product(*param_grid.values())]

        for params in param_combinations:
            model = OPTICS(**params)
            labels = model.fit_predict(self.X)

            # Calculate clustering metrics
            metrics = self._calculate_metrics(self.X, labels)

            # Store results
            result = {
                'params': params,
                'labels': labels,
                'metrics': metrics
            }
            self.results.append(result)

            # Update best model based on silhouette score
            if metrics['silhouette'] > self.best_score:
                self.best_score = metrics['silhouette']
                self.best_params = params
                self.best_model = model

    def _calculate_metrics(self, X, labels):
        """Calculate various clustering metrics."""
        # Filter out noise points (label -1) for metric calculation
        mask = labels != -1
        if sum(mask) < 2:  # Need at least 2 points for metrics
            return {
                'silhouette': float('-inf'),
                'calinski_harabasz': float('-inf'),
                'davies_bouldin': float('inf')
            }

        try:
            sil_score = silhouette_score(X[mask], labels[mask]) if len(set(labels[mask])) > 1 else float('-inf')
            ch_score = calinski_harabasz_score(X[mask], labels[mask]) if len(set(labels[mask])) > 1 else float('-inf')
            db_score = davies_bouldin_score(X[mask], labels[mask]) if len(set(labels[mask])) > 1 else float('inf')
        except:
            sil_score = float('-inf')
            ch_score = float('-inf')
            db_score = float('inf')

        return {
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        }

    def plot_results(self, n_best=3):
        """
        Plot clustering results for the n best parameter combinations.

        Parameters:
        n_best (int): Number of best results to plot
        """
        # Sort results by silhouette score
        sorted_results = sorted(self.results,
                                key=lambda x: x['metrics']['silhouette'],
                                reverse=True)[:n_best]

        n_plots = len(sorted_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for ax, result in zip(axes, sorted_results):
            labels = result['labels']

            # Create scatter plot
            scatter = ax.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis')
            ax.set_title(f"Silhouette: {result['metrics']['silhouette']:.3f}\n"
                         f"min_samples: {result['params']['min_samples']}\n"
                         f"max_eps: {result['params']['max_eps']}\n"
                         f"xi: {result['params']['xi']}")

        plt.tight_layout()
        plt.show()

    def get_best_results(self):
        """Return the best clustering results and parameters."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_labels': self.best_model.labels_ if self.best_model else None,
            'all_results': self.results
        }


# Example usage and testing
def test_optics_analyzer():
    df = pd.read_csv("../Visual_datasets/spiral.csv")

    # Initialize and run analyzer
    analyzer = OpticsClusterAnalyzer(df, label_column='Class')

    # Define custom parameter grid
    param_grid = {
        'min_samples': [5, 10],
        'max_eps': [np.inf, 2.0],
        'min_cluster_size': [5, 10],
        'xi': [0.05, 0.1]
    }

    # Perform grid search
    analyzer.grid_search(param_grid)

    # Plot results
    analyzer.plot_results(n_best=2)

    # Get best results
    best_results = analyzer.get_best_results()
    print("\nBest Parameters:", best_results['best_params'])
    print("Best Silhouette Score:", best_results['best_score'])


if __name__ == "__main__":
    test_optics_analyzer()