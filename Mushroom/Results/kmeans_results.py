import os
import time
import pandas as pd
import numpy as np
from Classes.KMeans import KMeansAlgorithm
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
repetitions = 10

# Initialize results DataFrame
results = []

# Perform tests
for k in range(2, max_k_value+1):
    for _ in range(repetitions):
        # Initialize random centroids by choosing k random samples
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]

        for distance_metric in distance_metrics:

            # Instantiate KMeans and measure performance
            start_time = time.time()
            kmeans = KMeansAlgorithm(k, centroids.copy(), distance_metric, max_iter)
            cluster_labels, E = kmeans.fit(X)
            end_time = time.time()

            # Evaluate clustering performance
            metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)
            execution_time = end_time - start_time

            algorithm = f'KMeans({k}, {distance_metric})'
            results.append({
                'Algorithm': algorithm,
                'E': E,
                **metrics,
                'Time': execution_time
            })

# Save results to CSV file
results_df = pd.DataFrame(results)
csv_path = os.path.join(dataset_path, "Results", "CSVs", "kmeans_results.csv")
results_df.to_csv(csv_path, index=False)