import os
import time
import pandas as pd
from Classes.GlobalKMeans import GlobalKMeansAlgorithm
from Classes.EvaluationUtils import EvaluationUtils

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Mushroom'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "mushroom.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0','Class']).values

# Define configurations to test
max_k_value = 20
distance_metrics = ['euclidean', 'manhattan', 'clark']
max_iter = 10

# Initialize results DataFrame
results = []

test_time = time.time()
# Perform tests
for k in range(2, max_k_value+1):

    n_buckets_dict = {'2k': 2*k,'3k': 3*k,'4k': 4*k}

    for n_buckets_str, n_buckets in n_buckets_dict.items():
        for distance_metric in distance_metrics:

            # Instantiate KMeans and measure performance
            start_time = time.time()
            global_kmeans = GlobalKMeansAlgorithm(k, distance_metric, max_iter, n_buckets)
            cluster_labels, E = global_kmeans.fit(X)
            end_time = time.time()

            # Evaluate clustering performance
            metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)
            execution_time = end_time - start_time

            algorithm = f'GlobalKMeans({k}, {distance_metric}, {n_buckets})'
            results.append({
                'Algorithm': algorithm,
                'k': k,
                'Distance_Metric': distance_metric,
                'N_Buckets': n_buckets_str,
                'E': E,
                **metrics,
                'Time': execution_time
            })
print("Total test time was: ", time.time() - test_time)

# Save results to CSV file
results_df = pd.DataFrame(results)
csv_path = os.path.join(dataset_path, "Results", "CSVs", "global_kmeans_results.csv")
results_df.to_csv(csv_path, index=False)