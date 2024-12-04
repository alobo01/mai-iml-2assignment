import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from Classes.EvaluationUtils import EvaluationUtils

def clustering(X, metric, algorithm):
    """
    Perform OPTICS clustering with specified metric and algorithm
    """
    optics = OPTICS(
        metric=metric,
        algorithm=algorithm,
        min_samples=20,  # Allows reasonable density requirement for smaller clusters
        xi=0.02,  # Controls steepness for identifying clusters
        min_cluster_size=0.02  # Ensures clusters with a reasonable number of mushrooms
    )

    return optics.fit(X), optics

# Load data
dataset_path = '..'
data_path = os.path.join(dataset_path, "Preprocessing", "mushroom.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0', 'Class']).values

# Test configurations
distances = ['euclidean', 'manhattan', 'cosine']
algorithm = 'brute'

# Color palette
colors = ['blue', 'red', 'green']

# Create figure
plt.figure(figsize=(12, 6))

# Perform clustering and plot reachability
results = []

for dist, color in zip(distances, colors):
    start_time = time.time()
    optics_model, full_optics = clustering(X, dist, algorithm)
    end_time = time.time()
    cluster_pred = optics_model.labels_
    
    # Evaluate clustering
    metrics = EvaluationUtils.evaluate(X, class_labels, cluster_pred)
    current_algorithm = f'Optics_{dist}_{algorithm}'
    execution_time = end_time - start_time
    
    # Plot reachability 
    plt.plot(full_optics.reachability_[full_optics.ordering_], 
             label=f'{dist} distance', 
             color=color, 
             alpha=0.7)
    
    # Collect results
    result_entry = {
        'Algorithm': current_algorithm,
        **metrics,
        'Time': execution_time
    }
    results.append(result_entry)

plt.title('Reachability Plots: Brute Algorithm (Mushroom Dataset)', fontsize=14)
plt.xlabel('Points (ordered)', fontsize=12)
plt.ylabel('Reachability Distance', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save plot
plot_path = os.path.join(dataset_path, 'Analysis/plots_and_tables/OPTICS/reachability_plots.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Save results
results_df = pd.DataFrame(results)
csv_path = os.path.join(dataset_path, 'Results/CSVs/optics_results.csv')
results_df.to_csv(csv_path, index=False)

print(f"Reachability plot saved to {plot_path}")
print(f"Results saved to {csv_path}")
