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
                        distance_metric=distance_metric,
                        max_iter=300  # KMeans internal max iterations
                    )

                    cluster_labels = xmeans.fit(X)
                    end_time = time.time()

                    # Check if the algorithm returned only one cluster
                    if xmeans.n_clusters == 1:
                        results.append({
                            'Algorithm': f'XMeans(max_k={max_clusters}, max_iter={max_iterations}, metric={distance_metric})',
                            'Actual_Clusters': xmeans.n_clusters,
                            'Max_Clusters': max_clusters,
                            'Max_Iterations': max_iterations,
                            'Distance_Metric': distance_metric,
                            'Repetition': rep + 1,
                            'E': np.nan,  # Not applicable
                            'Time': end_time - start_time,
                            'silhouette_score': np.nan,
                            'calinski_harabasz_score': np.nan,
                            'davies_bouldin_score': np.nan,
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
                        'Algorithm': f'XMeans(max_k={max_clusters}, max_iter={max_iterations}, metric={distance_metric})',
                        'Actual_Clusters': xmeans.n_clusters,
                        'Max_Clusters': max_clusters,
                        'Max_Iterations': max_iterations,
                        'Distance_Metric': distance_metric,
                        'Repetition': rep + 1,
                        'E': E,
                        'Time': execution_time,
                        **metrics,
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
max_clusters_list = [5, 10, 15, 20]
max_iterations_list = [10, 50, 100]
distance_metrics = ['euclidean', 'manhattan', 'clark']
repetitions = 3

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
summary_stats = results_df.groupby(['Max_Clusters', 'Max_Iterations', 'Distance_Metric']).agg({
    'Actual_Clusters': 'mean',
    'E': 'mean',
    'Time': ['mean', 'std'],
    'silhouette_score': 'mean',
    'calinski_harabasz_score': 'mean',
    'davies_bouldin_score': 'mean'
}).round(3)

print(summary_stats)

# Find best configurations based on different metrics
print("\nBest Configurations:")


# Safely handle NaN values and empty results
def get_best_configuration(df, metric, find_max=True):
    """Find the best configuration based on a metric."""
    filtered_df = df[df[metric].notna()]
    if filtered_df.empty:
        return None  # No valid rows for this metric
    if find_max:
        return filtered_df.loc[filtered_df[metric].idxmax()]
    else:
        return filtered_df.loc[filtered_df[metric].idxmin()]


# Determine the best configurations
best_silhouette = get_best_configuration(results_df, 'silhouette_score', find_max=True)
best_calinski = get_best_configuration(results_df, 'calinski_harabasz_score', find_max=True)
best_davies = get_best_configuration(results_df, 'davies_bouldin_score', find_max=False)
best_variance = get_best_configuration(results_df, 'E', find_max=False)

# Print results if available
if best_silhouette is not None:
    print("\nBest by Silhouette Score:")
    print(best_silhouette[['Algorithm', 'Actual_Clusters', 'silhouette_score', 'Time']])
else:
    print("\nNo valid configurations found for Silhouette Score.")

if best_calinski is not None:
    print("\nBest by Calinski-Harabasz Score:")
    print(best_calinski[['Algorithm', 'Actual_Clusters', 'calinski_harabasz_score', 'Time']])
else:
    print("\nNo valid configurations found for Calinski-Harabasz Score.")

if best_davies is not None:
    print("\nBest by Davies-Bouldin Score:")
    print(best_davies[['Algorithm', 'Actual_Clusters', 'davies_bouldin_score', 'Time']])
else:
    print("\nNo valid configurations found for Davies-Bouldin Score.")

if best_variance is not None:
    print("\nBest by Total Variance (E):")
    print(best_variance[['Algorithm', 'Actual_Clusters', 'E', 'Time']])
else:
    print("\nNo valid configurations found for Total Variance (E).")
