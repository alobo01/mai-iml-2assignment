from Classes.ResultUtils import ResultUtils
from Classes.XMeans import XMeans
import os
import pandas as pd

if __name__ == "__main__" or "__mp_main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load dataset
data_path = os.path.join(dataset_path, "Preprocessing", "hepatitis.csv")
data = pd.read_csv(data_path, index_col=0)
class_labels = data['Class']
X = data.drop(columns=['Class']).values


# Define configurations to test
grid = {
    'max_clusters': [x for x in range(2, 11)],
    'Repetitions': 10
}

# File paths for saving results
results_file = os.path.join(dataset_path, "Results", "CSVs", "xmeans_results.csv")
labels_file = os.path.join(dataset_path, "Results", "CSVs", "xmeans_cluster_labels.csv")

ResultUtils.runGrid(grid, XMeans, X, class_labels, results_file, labels_file)