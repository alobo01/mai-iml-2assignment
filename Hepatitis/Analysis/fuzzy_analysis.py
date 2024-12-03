from Classes.ViolinPlotsUtils import ViolinPlotter
import os
import pandas as pd


if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load the FCM results
csv_path = os.path.join(dataset_path, "Results", "CSVs", "fuzzy_c_means_results.csv")
results_df = pd.read_csv(csv_path)

vp = ViolinPlotter(results_df)
vp.create_violin_plot("n_clusters","DBI",add_jitter=False)
vp.create_violin_plot("fuzziness","DBI",add_jitter=False)