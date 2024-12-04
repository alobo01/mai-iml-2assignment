from Classes.ResultUtils import ResultUtils
from Classes.SpectralClustering import SpectralClusteringWrapper
import os
import pandas as pd

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "hepatitis.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0','Class']).values

# Define grid
grid = {
    'n_clusters': [2],
    'n_neighbors': [5, 7, 10],
    'assign_labels': ['kmeans', 'cluster_qr'],
    'eigen_solver': ['lobpcg', 'amg', 'arpack'],
    'Repetitions': 10
}

# File paths
results_file = os.path.join(dataset_path, 'Results/CSVs/spectral_clustering_results.csv')
labels_file = os.path.join(dataset_path, 'Results/CSVs/spectral_clustering_cluster_labels.csv')

# Run grid search and save results
ResultUtils.runGrid(grid, SpectralClusteringWrapper, X, class_labels, results_file, labels_file)