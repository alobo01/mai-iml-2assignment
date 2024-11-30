import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict
from Classes.EvaluationUtils import EvaluationUtils
from Classes.KMeans import KMeansAlgorithm
from Classes.XMeans import XMeans


def test_xmeans_configurations(
        X: np.ndarray,
        class_labels: np.ndarray,
        max_clusters_list: List[int],
        max_iterations_list: List[int],
        distance_metrics: List[str],
        repetitions: int = 1
) -> pd.DataFrame:
    """
    Test XMeans with different configurations and evaluate performance.

    Args:
        X: Input data
        class_labels: True class labels
        max_clusters_list: List of maximum clusters to test
        max_iterations_list: List of maximum iterations to test
        distance_metrics: List of distance metrics to test
        repetitions: Number of times to repeat each configuration

    Returns:
        DataFrame with results
    """
    results = []

    for max_clusters in max_clusters_list:
        for max_iterations in max_iterations_list:
            for distance_metric in distance_metrics:
                for rep in range(repetitions):
                    # Initialize and run XMeans
                    start_time = time.time()

                    xmeans = XMeans(
                        max_clusters=max_clusters,
                        max_iterations=max_iterations,
                        distance_metric=distance_metric
                    )

                    cluster_labels = xmeans.fit(X)
                    end_time = time.time()

                    # Check if the algorithm returned only one cluster
                    if xmeans.n_clusters == 1:
                        results.append({
                            'Algorithm': f'XMeans({max_clusters}, {distance_metric})',# , {max_iterations})',
                            'Max_Clusters': max_clusters,
                            'Distance_Metric': distance_metric,
                            # 'Max_Iterations': max_iterations,
                            'Actual_Clusters': xmeans.n_clusters,
                            'Repetition': rep + 1,
                            'E': np.nan,  # Not applicable
                            'ARI': np.nan,
                            'F1 Score': np.nan,
                            'DBI': np.nan,
                            'silhouette_score': np.nan,
                            'calinski_harabasz_score': np.nan,
                            'Time': end_time - start_time,
                            'Status': 'Fail (1 Cluster)'
                        })
                        continue

                    # Calculate total variance (E)
                    kmeans_final = KMeansAlgorithm(
                        k=xmeans.n_clusters,
                        centroids=xmeans.centroids,
                        distance_metric=distance_metric
                    )
                    _, E = kmeans_final.fit(X)

                    # Evaluate clustering performance
                    metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)
                    execution_time = end_time - start_time

                    # Record results
                    results.append({
                        'Algorithm': f'XMeans({max_clusters}, {distance_metric})',# , {max_iterations})',
                        'Max_Clusters': max_clusters,
                        'Distance_Metric': distance_metric,
                        # 'Max_Iterations': max_iterations,
                        'Actual_Clusters': xmeans.n_clusters,
                        'Repetition': rep + 1,
                        'E': E,
                        **metrics,
                        'Time': execution_time,
                        'Status': 'Success'
                    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Set paths
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing/hepatitis.csv")
data = pd.read_csv(data_path, index_col=0)
class_labels = data['Class']
X = data.drop(columns=['Class']).values

# Define configurations to test
max_clusters_list = [k for k in range(2, 21)]
distance_metrics = ['euclidean', 'manhattan', 'clark']
max_iterations_list = [50]
repetitions = 10

# Run tests
results_df = test_xmeans_configurations(
    X=X,
    class_labels=class_labels,
    max_clusters_list=max_clusters_list,
    max_iterations_list=max_iterations_list,
    distance_metrics=distance_metrics,
    repetitions=repetitions
)

# Save results
results_path = os.path.join(dataset_path, 'Results/CSVs/xmeans_results.csv')
os.makedirs(os.path.dirname(results_path), exist_ok=True)
results_df.to_csv(results_path, index=False)

# Print summary statistics
print("\nSummary Statistics:")
summary_stats = results_df.groupby(['Max_Clusters', 'Distance_Metric']).agg({ # 'Max_Iterations', 'Distance_Metric']).agg({
    'Actual_Clusters': 'mean',
    'E': 'min',
    'Time': ['mean', 'std'],
    'silhouette_score': 'max',
    'calinski_harabasz_score': 'max',
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

    if 'DBI' or 'E' in metric_column:
        # For Davies-Bouldin, lower is better
        best_idx = valid_rows[metric_column].idxmin()
    else:
        # For other metrics, higher is better
        best_idx = valid_rows[metric_column].idxmax()

    return valid_rows.loc[best_idx]


# Calculate best configurations
best_silhouette = get_best_configuration(results_df, 'silhouette_score')
best_calinski = get_best_configuration(results_df, 'calinski_harabasz_score')
best_davies = get_best_configuration(results_df, 'DBI')
best_variance = get_best_configuration(results_df, 'E')


# Print the best configurations
def print_best_config(config, metric_name):
    if config is not None:
        print(f"\nBest by {metric_name}:")
        print(config[['Algorithm', 'Actual_Clusters', metric_name, 'Time']])


print_best_config(best_silhouette, 'silhouette_score')
print_best_config(best_calinski, 'calinski_harabasz_score')
print_best_config(best_davies, 'DBI')
print_best_config(best_variance, 'E')