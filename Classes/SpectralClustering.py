import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from itertools import product


class SpectralClusterAnalyzer:
    def __init__(self, df, label_column=None):
        """
        Initialize the Spectral Clustering analyzer.

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

    def _estimate_n_clusters(self):
        """Estimate a reasonable range for n_clusters using eigenvalue analysis."""
        # Create affinity matrix using RBF kernel
        affinity = kneighbors_graph(self.X, n_neighbors=10, mode='distance').toarray()
        affinity = np.exp(-affinity ** 2 / (2. * np.std(affinity) ** 2))

        # Compute eigenvalues
        eigenvalues = np.sort(np.linalg.eigvals(affinity))[::-1]

        # Find the elbow point in eigenvalue curve
        diffs = np.diff(eigenvalues)
        elbow = np.argmax(diffs) + 1

        # Return a reasonable range around the elbow point
        min_clusters = max(2, elbow - 2)
        max_clusters = min(len(eigenvalues) - 1, elbow + 3)

        return list(range(min_clusters, max_clusters + 1))

    def grid_search(self, param_grid=None):
        """
        Perform grid search over specified parameters.

        Parameters:
        param_grid (dict): Dictionary of parameters to try. If None, uses default grid.
        """
        if param_grid is None:
            # Estimate reasonable n_clusters range
            n_clusters_range = self._estimate_n_clusters()

            param_grid = {
                'n_clusters': n_clusters_range,
                'assign_labels': ['kmeans', 'discretize'],
                'affinity': ['rbf', 'nearest_neighbors'],
                'n_neighbors': [5, 10, 15],  # Only used when affinity='nearest_neighbors'
                'gamma': [0.1, 1.0, 10.0]  # Only used when affinity='rbf'
            }

        # Generate all valid parameter combinations
        param_combinations = []
        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            # Skip invalid combinations
            if params['affinity'] == 'rbf' and 'n_neighbors' in params:
                del params['n_neighbors']
            if params['affinity'] == 'nearest_neighbors' and 'gamma' in params:
                del params['gamma']
            param_combinations.append(params)

        for params in param_combinations:
            try:
                model = SpectralClustering(random_state=42, **params)
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

            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue

    def _calculate_metrics(self, X, labels):
        """Calculate various clustering metrics."""
        try:
            sil_score = silhouette_score(X, labels) if len(set(labels)) > 1 else float('-inf')
            ch_score = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else float('-inf')
            db_score = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float('inf')
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

            # Create title with key parameters and metrics
            title = f"Silhouette: {result['metrics']['silhouette']:.3f}\n"
            title += f"n_clusters: {result['params']['n_clusters']}\n"
            title += f"affinity: {result['params']['affinity']}\n"
            title += f"assign_labels: {result['params']['assign_labels']}"

            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def plot_eigenvalue_analysis(self):
        """Plot eigenvalue analysis for cluster number estimation."""
        # Create affinity matrix
        affinity = kneighbors_graph(self.X, n_neighbors=10, mode='distance').toarray()
        affinity = np.exp(-affinity ** 2 / (2. * np.std(affinity) ** 2))

        # Compute eigenvalues
        eigenvalues = np.sort(np.linalg.eigvals(affinity))[::-1]

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Analysis for Optimal Cluster Number')
        plt.grid(True)
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
def test_spectral_analyzer():
    # Generate sample data
    np.random.seed(42)
    n_samples = 300

    # Create three non-linearly separable clusters
    t = np.linspace(0, 2 * np.pi, n_samples // 3)
    # First cluster: circle
    circle_x = np.cos(t)
    circle_y = np.sin(t)
    cluster1 = np.column_stack([circle_x, circle_y]) + np.random.normal(0, 0.1, (n_samples // 3, 2))

    # Second and third clusters: half moons
    cluster2 = np.column_stack([np.cos(t), np.sin(t)]) * 2 + np.array([2, 2])
    cluster3 = np.column_stack([np.cos(t), -np.sin(t)]) * 2 + np.array([2, -2])
    cluster2 += np.random.normal(0, 0.1, (n_samples // 3, 2))
    cluster3 += np.random.normal(0, 0.1, (n_samples // 3, 2))

    # Combine clusters
    X = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.repeat([0, 1, 2], n_samples // 3)

    # Create dataframe
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['label'] = true_labels

    # Initialize and run analyzer
    analyzer = SpectralClusterAnalyzer(df, label_column='label')

    # Plot eigenvalue analysis
    analyzer.plot_eigenvalue_analysis()

    # Define custom parameter grid
    param_grid = {
        'n_clusters': [3, 4, 5],
        'assign_labels': ['kmeans'],
        'affinity': ['rbf', 'nearest_neighbors'],
        'n_neighbors': [5, 10, 50, 100],
        'gamma': [0.1, 1.0]
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
    test_spectral_analyzer()