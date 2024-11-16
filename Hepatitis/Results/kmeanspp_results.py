import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import cluster, adjusted_rand_score, f1_score, davies_bouldin_score
from Classes.KMeansPP import KMeansPPAlgorithm

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing/hepatitis.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0','Class']).values

# Define configurations to test
k_values = [2, 4, 6, 8]
distance_metrics = ['euclidean', 'manhattan', 'clark']
max_iter = 1

# Initialize results DataFrame
results = []

# Perform tests
for k in k_values:
    for _ in range(3):
        for distance_metric in distance_metrics:

            # Instantiate KMeans and measure performance
            start_time = time.time()
            kmeans = KMeansPPAlgorithm(k, distance_metric, max_iter)
            cluster_labels = kmeans.fit(X)
            end_time = time.time()

            ari = adjusted_rand_score(class_labels, cluster_labels)
            fm = f1_score(class_labels, cluster_labels, average='macro')
            dbi = davies_bouldin_score(X, cluster_labels)
            purity = purity_score(class_labels, cluster_labels)
            execution_time = end_time - start_time

            algorithm = f'KMeansPP({k}, {distance_metric})'
            results.append({
                'Algorithm': algorithm,
                'ARI': ari,
                'Fm': fm,
                'DBI': dbi,
                'Purity': purity,
                'Time': execution_time
            })

# Save results to CSV file
results_df = pd.DataFrame(results)
csv_path = os.path.join(dataset_path, 'Results/CSVs/kmeanspp_results.csv')
results_df.to_csv(csv_path, index=False)