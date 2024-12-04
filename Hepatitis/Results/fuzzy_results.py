import os
import pandas as pd
from Classes.FuzzyClustering import FuzzyCMeans
from Classes.ResultUtils import ResultUtils

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "hepatitis.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0', 'Class']).values

# Define configurations to test
grid = {
    'n_clusters': [6, 8, 9, 10],
    'fuzziness': [1.5, 2, 3, 4],  # Fuzziness parameter 'm'
    'max_iter': [100, 300, 500],  # Maximum iterations
    'error': [1e-1, 1e-4, 1e-5],  # Convergence tolerance
    'rho': [0.5, 0.7, 0.9],  # rho parameter
    'Repetitions': 10  # Number of repetitions
}

# File paths for saving results
results_file = os.path.join(dataset_path, "Results", "CSVs", "fuzzy_c_means_results.csv")
labels_file = os.path.join(dataset_path, "Results", "CSVs", "fuzzy_c_means_cluster_labels.csv")

# Run grid search and save results
ResultUtils.runGrid(grid, FuzzyCMeans, X, class_labels, results_file, labels_file)
