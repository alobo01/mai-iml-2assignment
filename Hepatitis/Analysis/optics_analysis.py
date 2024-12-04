import pandas as pd
import os
from Classes.AnalysisUtils import AnalysisUtils

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load the K-Means results
results_path = os.path.join(dataset_path, "Results", "CSVs", "optics_clustering_results.csv")
results_df = pd.read_csv(results_path)

cluster_labels_path = os.path.join(dataset_path, "Results", "CSVs", "optics_clustering_cluster_labels.csv")
labels_df = pd.read_csv(cluster_labels_path)

pca_dataset_path = os.path.join(dataset_path, "Preprocessing", "hepatitis_pca.csv")
pca_dataset_df = pd.read_csv(pca_dataset_path)

# Create output directories
base_path = 'plots_and_tables'
plots_path = os.path.join(base_path, 'OPTICS')

# Ensure output directories exist
os.makedirs(plots_path, exist_ok=True)

features_explored = ['metric', 'algorithm']

AnalysisUtils.totalAnalysis(results_df, labels_df, pca_dataset_df, plots_path, features_explored)

print("Optics clustering analysis completed successfully.")
print("Output files are available in:", base_path)