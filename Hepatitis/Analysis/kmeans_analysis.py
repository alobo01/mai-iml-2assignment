import pandas as pd
import os
from Classes.AnalysisUtils import AnalysisUtils

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load the K-Means results
results_path = os.path.join(dataset_path, "Results", "CSVs", "kmeans_results.csv")
results_df = pd.read_csv(results_path)

cluster_labels_path = os.path.join(dataset_path, "Results", "CSVs", "kmeans_cluster_labels.csv")
labels_df = pd.read_csv(cluster_labels_path)

# Create output directories
base_path = 'plots_and_tables'
plots_path = os.path.join(base_path, 'KMeansPlots')

# Ensure output directories exist
os.makedirs(plots_path, exist_ok=True)

# Metrics to analyze
metrics = ['ARI', 'NMI', 'DBI', 'Silhouette', 'CHS', 'Time']

# 1. Create Pairplot for Hyperparameter Analysis
AnalysisUtils.create_pairplot(
    data=results_df,
    params=['k', 'Distance_Metric'],
    metric='ARI',  # Using ARI as primary performance metric
    agg_func='max',
    plots_path=plots_path
)

# 2. Create Custom Heatmap for Metric Correlations
AnalysisUtils.plot_custom_heatmap(results_df[metrics], plots_path=plots_path)

# 3.
best_runs = AnalysisUtils.plot_best_runs(results_df, labels_df)
print(best_runs)

print("K-Means clustering analysis completed successfully.")
print("Output files are available in:", base_path)