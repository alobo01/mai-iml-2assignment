import os
import time
import pandas as pd
import numpy as np
from typing import List
from Classes.EvaluationUtils import EvaluationUtils
from Classes.XMeans import XMeans


def test_xmeans_tolerance(
        X: np.ndarray,
        class_labels: np.ndarray,
        tolerance_values: List[float],
        max_clusters: int,
        max_iterations: int,
        repetitions: int = 1
) -> pd.DataFrame:
    """
    Test XMeans with different tolerance parameters and evaluate performance.

    Args:
        X: Input data
        class_labels: True class labels
        tolerance_values: List of tolerance values to test
        max_clusters: Maximum number of clusters
        max_iterations: Maximum number of iterations
        repetitions: Number of times to repeat each configuration

    Returns:
        DataFrame with results
    """
    results = []

    for tol in tolerance_values:
        for rep in range(repetitions):
            # Initialize and run XMeans
            start_time = time.time()

            xmeans = XMeans(
                max_clusters=max_clusters,
                max_iterations=max_iterations,
                tol=tol
            )

            cluster_labels = xmeans.fit(X)
            end_time = time.time()

            # Check if the algorithm returned only one cluster
            if xmeans.n_clusters == 1:
                results.append({
                    'Algorithm': f'XMeans(Tol={tol})',
                    'Tolerance': tol,
                    'Max_Clusters': max_clusters,
                    'Actual_Clusters': xmeans.n_clusters,
                    'Repetition': rep + 1,
                    'E': np.nan,  # Not applicable
                    'ARI': np.nan,
                    'F1 Score': np.nan,
                    'DBI': np.nan,
                    'Silhouette_Score': np.nan,
                    'Calinski_Harabasz_Score': np.nan,
                    'Time': end_time - start_time,
                    'Status': 'Fail (1 Cluster)'
                })
                continue

            # Calculate total variance (E)
            metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)
            execution_time = end_time - start_time

            # Record results
            results.append({
                'Algorithm': f'XMeans(Tol={tol})',
                'Tolerance': tol,
                'Max_Clusters': max_clusters,
                'Actual_Clusters': xmeans.n_clusters,
                'Repetition': rep + 1,
                'E': metrics.get('E', np.nan),
                **metrics,
                'Time': execution_time,
                'Status': 'Success'
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Set paths
    dataset_path = '..'

    # Load dataset
    data_path = os.path.join(dataset_path, "Preprocessing/pen-based.csv")
    data = pd.read_csv(data_path, index_col=0)
    class_labels = data['Class']
    X = data.drop(columns=['Class']).values

    # Define configurations to test
    tolerance_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    max_clusters = 50
    max_iterations = 1000
    repetitions = 10

    # Run tests
    results_df = test_xmeans_tolerance(
        X=X,
        class_labels=class_labels,
        tolerance_values=tolerance_values,
        max_clusters=max_clusters,
        max_iterations=max_iterations,
        repetitions=repetitions
    )

    # Save results
    results_path = os.path.join(dataset_path, 'Results/CSVs/xmeans_results.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)

    # Print summary statistics
    print("\nSummary Statistics:")
    summary_stats = results_df.groupby(['Tolerance']).agg({
        'Actual_Clusters': 'mean',
        'E': 'min',
        'Time': ['mean', 'std'],
        'Silhouette_Score': 'max',
        'Calinski_Harabasz_Score': 'max',
        'DBI': 'min'
    }).round(3)

    print(summary_stats)

    # Find best configurations based on different metrics
    print("\nBest Configurations:")


    # Safely handle NaN values before finding the best configurations
    def get_best_configuration(df, metric_column):
        valid_rows = df[df[metric_column].notna()]
        if valid_rows.empty:
            print(f"No valid rows found for {metric_column}")
            return None

        if metric_column in ['DBI', 'E']:
            # For Davies-Bouldin, lower is better
            best_idx = valid_rows[metric_column].idxmin()
        else:
            # For other metrics, higher is better
            best_idx = valid_rows[metric_column].idxmax()

        return valid_rows.loc[best_idx]


    # Calculate best configurations
    best_silhouette = get_best_configuration(results_df, 'Silhouette_Score')
    best_calinski = get_best_configuration(results_df, 'Calinski_Harabasz_Score')
    best_davies = get_best_configuration(results_df, 'DBI')
    best_variance = get_best_configuration(results_df, 'E')

    # Print the best configurations
    def print_best_config(config, metric_name):
        if config is not None:
            print(f"\nBest by {metric_name}:")
            print(config[['Algorithm', 'Actual_Clusters', metric_name, 'Time']])


    print_best_config(best_silhouette, 'Silhouette_Score')
    print_best_config(best_calinski, 'Calinski_Harabasz_Score')
    print_best_config(best_davies, 'DBI')
    print_best_config(best_variance, 'E')
