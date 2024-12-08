from Classes.ResultUtils import ResultUtils
from Classes.OpticsClustering import OPTICSClusteringWrapper
import os
import pandas as pd

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Mushroom'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "mushroom.csv")
data = pd.read_csv(data_path, index_col=0)
class_labels = data['Class']
X = data.drop(columns=['Class']).values

# Define grid
grid = {
    'metric': ['euclidean', 'manhattan', 'chebyshev'],
    'algorithm': ['brute', 'ball_tree', 'kd_tree'],
    'xi': [0.1],
    'min_cluster_size': [0.1],
    'min_samples': [10],
    'Repetitions': 1
}

# File paths
results_file = os.path.join(dataset_path, 'Results/CSVs/optics_clustering_results.csv')
labels_file = os.path.join(dataset_path, 'Results/CSVs/optics_clustering_cluster_labels.csv')

# Run grid search and save results
ResultUtils.runGrid(grid, OPTICSClusteringWrapper, X, class_labels, results_file, labels_file)


