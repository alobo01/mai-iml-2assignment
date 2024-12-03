import os
import time
import pandas as pd
from multiprocessing import Pool

from Classes.FuzzyClustering import FuzzyCMeans
from Classes.EvaluationUtils import EvaluationUtils

if __name__ == "__main__" or "__mp_main__":
    dataset_path = '..'
else:
    dataset_path = 'Mushroom'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "mushroom.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0', 'Class']).values

# Define configurations to test
grid = {
    'n_clusters': [6, 8, 9, 10],
    'fuzziness': [1.5, 2, 3, 4],  # Fuzziness parameter 'm'
    'max_iter': [100, 300, 500],  # Maximum iterations
    'epsilon': [1e-1, 1e-4, 1e-5],  # Convergence tolerance
    'rho': [0.5, 0.7, 0.9],  # rho parameter
    'n_init': 10  # Number of repetitions
}

# Define a function to evaluate a single configuration
def evaluate_config(task):
    n_clusters, fuzziness, max_iter, epsilon, rho, init_index = task
    try:
        start_time = time.time()

        # Instantiate Fuzzy C-Means with the given parameters
        fuzzy_c_means = FuzzyCMeans(
            n_clusters=n_clusters,
            m=fuzziness,
            max_iter=max_iter,
            error=epsilon,
            rho=rho
        )
        fuzzy_c_means.fit(X)
        cluster_labels = fuzzy_c_means.predict(X)
        end_time = time.time()

        # Evaluate clustering performance
        metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)
        execution_time = end_time - start_time
        print(f"Done ({n_clusters}, {fuzziness}, {max_iter}, {epsilon}, {rho}, {init_index})")
        # Return the results including hyperparameters and n_init index
        return {
            'Algorithm': "FuzzyCMeans",
            'n_clusters': n_clusters,
            'fuzziness': fuzziness,
            'max_iter': max_iter,
            'epsilon': epsilon,
            'rho': rho,
            'n_init_index': init_index,
            **metrics,
            'Time': execution_time
        }
    except Exception as e:
        print(f"Error with configuration: {n_clusters}, {fuzziness}, {max_iter}, {epsilon}, {rho}, init_index: {init_index}. Error: {e}")
        return None

# Generate all configurations
def generate_configs():
    tasks = []
    n_init = grid['n_init']
    for n_clusters in grid['n_clusters']:
        for fuzziness in grid['fuzziness']:
            for max_iter in grid['max_iter']:
                for epsilon in grid['epsilon']:
                    for rho in grid['rho']:
                        for init_index in range(n_init):
                            tasks.append((n_clusters, fuzziness, max_iter, epsilon, rho, init_index))
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
    csv_path = os.path.join(dataset_path, 'Results/CSVs/fuzzy_c_means_results.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")
