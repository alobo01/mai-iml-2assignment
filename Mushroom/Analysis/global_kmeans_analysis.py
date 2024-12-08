import pandas as pd
import os
from Classes.AnalysisUtils import AnalysisUtils

if __name__ == "__main__":
    dataset_path = '..'
    # Create output directories
    base_path = 'plots_and_tables'
else:
    dataset_path = 'Mushroom'
    # Create output directories
    base_path = os.path.join(dataset_path,'Analysis','plots_and_tables')

# Load the K-Means results
results_path = os.path.join(dataset_path, "Results", "CSVs", "global_kmeans_results.csv")
results_df = pd.read_csv(results_path)

cluster_labels_path = os.path.join(dataset_path, "Results", "CSVs", "global_kmeans_cluster_labels.csv")
labels_df = pd.read_csv(cluster_labels_path)

pca_dataset_path = os.path.join(dataset_path, "Preprocessing", "mushroom_pca.csv")
pca_dataset_df = pd.read_csv(pca_dataset_path)

umap_dataset_path = os.path.join(dataset_path, "Preprocessing", "mushroom_pca.csv")
umap_dataset_df = pd.read_csv(umap_dataset_path)


plots_path = os.path.join(base_path, 'GlobalKMeansPlots')

# Ensure output directories exist
os.makedirs(plots_path, exist_ok=True)

features_explored = ['k', 'Distance_Metric', 'N_Buckets']

AnalysisUtils.totalAnalysis(results_df, labels_df, pca_dataset_df, umap_dataset_df, plots_path, features_explored)

print("K-Means clustering analysis completed successfully.")
print("Output files are available in:", base_path)