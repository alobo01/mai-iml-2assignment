import os
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from Classes.SpectralClustering import SpectralClusteringWrapper
from Classes.EvaluationUtils import EvaluationUtils

# Memory leaks warning
os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == "__main__" or "__mp_main__":
    dataset_path = '..'
else:
    dataset_path = 'Pen-based'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing","pen-based.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0', 'Class']).values

# Define configurations to test
grid = {
    'n_clusters': [6, 8, 9, 10],
    'affinity': {
        'rbf': {'gamma': [0.1, 1, 10]},
        'nearest_neighbors': {'n_neighbors': [5, 10]}
    },
    'assign_labels': ['kmeans', 'discretize'],
    'n_init': [5, 10],
    'eigen_tol': [1e-3, 1e-4]
}

# Define a function to evaluate a single configuration
def evaluate_config(task):
    n_clusters, affinity, assign_labels, gamma, n_neighbors, n_init, eigen_tol = task
    try:
        start_time = time.time()

        # Instantiate SpectralClustering with the given parameters
        spectral_clustering = SpectralClusteringWrapper(
            n_clusters=n_clusters,
            affinity=affinity,
            assign_labels=assign_labels,
            gamma=gamma,
            n_neighbors=n_neighbors,
            n_init=n_init,
            eigen_tol=eigen_tol
        )
        cluster_labels = spectral_clustering.fit(X)
        end_time = time.time()

        # Evaluate clustering performance
        metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)
        execution_time = end_time - start_time

        # Return the results
        return {
            'Algorithm': f"SpectralClustering({n_clusters}, {affinity}, {assign_labels}, "
                         f"{gamma if affinity == 'rbf' else 'N/A'}, "
                         f"{n_neighbors if affinity == 'nearest_neighbors' else 'N/A'}, "
                         f"{n_init}, {eigen_tol})",
            **metrics,
            'Time': execution_time
        }
    except Exception as e:
        print(f"Error with configuration: {n_clusters}, {affinity}, {assign_labels}, "
              f"{gamma}, {n_neighbors}, {n_init}, {eigen_tol}. Error: {e}")
        return None

# Generate all configurations
def generate_configs():
    tasks = []
    for n_clusters in grid['n_clusters']:
        for affinity, params in grid['affinity'].items():
            for assign_labels in grid['assign_labels']:
                for n_init in grid['n_init']:
                    for eigen_tol in grid['eigen_tol']:
                        if affinity == 'rbf':
                            for gamma in params['gamma']:
                                tasks.append((n_clusters, affinity, assign_labels, gamma, 1, n_init, eigen_tol))
                        elif affinity == 'nearest_neighbors':
                            for n_neighbors in params['n_neighbors']:
                                tasks.append((n_clusters, affinity, assign_labels, 1, n_neighbors, n_init, eigen_tol))
    return tasks

# Main execution
if __name__ == "__main__":
    # Generate all tasks
    tasks = generate_configs()

    # Use multiprocessing Pool with max 4 workers
    max_workers = 4
    with Pool(processes=max_workers) as pool:
        results = pool.map(evaluate_config, tasks)

    # Filter out failed runs
    results = [res for res in results if res is not None]

    # Save results to CSV file
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(dataset_path, 'Results/CSVs/spectral_clustering_results.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")
