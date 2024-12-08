from Classes.ResultUtils import ResultUtils
from Classes.GlobalKMeans import GlobalKMeansAlgorithm
import os
import pandas as pd

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Pen-based'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "pen-based.csv")
data = pd.read_csv(data_path, index_col=0)
class_labels = data['Class']
X = data.drop(columns=['Class']).values

# Define configurations to test
max_k_value = 20
grid = {
    'k': [k for k in range(2, max_k_value+1)],
    'Distance_Metric': ['euclidean', 'manhattan', 'clark'],
    'N_Buckets': ['2k', '3k', '4k'],
    'Repetitions': 1  # Number of repetitions
}

# File paths for saving results
results_file = os.path.join(dataset_path, "Results", "CSVs", "global_kmeans_results.csv")
labels_file = os.path.join(dataset_path, "Results", "CSVs", "global_kmeans_cluster_labels.csv")

# Run grid search and save results
ResultUtils.runGrid(grid, GlobalKMeansAlgorithm, X, class_labels, results_file, labels_file)