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
        max_iterations_list: List of maximum iterations to  test
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
    dataset_path = 'Pen-based'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing/pen-based.csv")
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
