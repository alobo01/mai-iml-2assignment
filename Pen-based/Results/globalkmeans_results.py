import os
import time
import pandas as pd
from Classes.GlobalKMeans import GlobalKMeansAlgorithm
from Classes.EvaluationUtils import EvaluationUtils

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Pen-based'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing/pen-based.csv")
data = pd.read_csv(data_path)
class_labels = data['a17']
X = data.drop(columns=['Unnamed: 0','a17']).values

# Define configurations to test
max_k_value = 3
distance_metrics = ['euclidean']#, 'manhattan', 'clark']
max_iter = 1
repetitions = 1

# Initialize results DataFrame
results = []

test_time = time.time()
# Perform tests
for k in range(2, max_k_value+1):
    for _ in range(repetitions):
        for distance_metric in distance_metrics:

            # Instantiate KMeans and measure performance
            start_time = time.time()
            global_kmeans = GlobalKMeansAlgorithm(k, distance_metric, max_iter)
            cluster_labels, E = global_kmeans.fit(X)
            end_time = time.time()

            # Evaluate clustering performance
            metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)
            execution_time = end_time - start_time

            algorithm = f'GlobalKMeans({k}, {distance_metric})'
            results.append({
                'Algorithm': algorithm,
                'E': E,
                **metrics,
                'Time': execution_time
            })
print("Total test time was: ", time.time() - test_time)

# Save results to CSV file
results_df = pd.DataFrame(results)
csv_path = os.path.join(dataset_path, 'Results/CSVs/global_kmeans_results.csv')
results_df.to_csv(csv_path, index=False)