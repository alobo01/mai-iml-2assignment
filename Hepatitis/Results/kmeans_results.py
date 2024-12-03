import os
import time
import pandas as pd
import numpy as np
from Classes.KMeans import KMeansAlgorithm
from Classes.EvaluationUtils import EvaluationUtils

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "hepatitis.csv")
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
labels_df = pd.DataFrame()

# Perform tests
for k in range(2, max_k_value+1):
    for repetition in range(repetitions):
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

            algorithm = f'KMeans({k}, {distance_metric})_{repetition}'
            results.append({
                'Algorithm': algorithm,
                'k': k,
                'Distance_Metric': distance_metric,
                'Repetition': repetition,
                **metrics,
                'Time': execution_time
            })

            labels_df = pd.concat([labels_df, pd.DataFrame({algorithm: cluster_labels})], axis=1)

# Save results to CSV files
results_df = pd.DataFrame(results)
results_path = os.path.join(dataset_path, "Results", "CSVs", "kmeans_results.csv")
results_df.to_csv(results_path, index=False)

cluster_labels_path = os.path.join(dataset_path, "Results", "CSVs", "kmeans_cluster_labels.csv")
labels_df.to_csv(cluster_labels_path, index=False)