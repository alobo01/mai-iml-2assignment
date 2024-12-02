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
        min_samples=10,  # Adjust based on dataset
        xi=0.05,
        min_cluster_size=0.05
    )

    return optics.fit(X), optics

# Load data
dataset_path = '..'
data_path = os.path.join(dataset_path, "Preprocessing", "hepatitis.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0', 'Class']).values

# Test configurations
distances = ['euclidean', 'manhattan', 'cosine']
algorithms = ['auto', 'brute', 'ball_tree', 'kd_tree']

# Perform clustering and generate reachability plots
results = []
plt.figure(figsize=(15, len(distances) * len(algorithms) * 3))
plot_index = 1

for dist in distances:
    if dist == 'cosine':
        algorithms = ['auto', 'brute']
    for algorithm in algorithms:
        start_time = time.time()
        optics_model, full_optics = clustering(X, dist, algorithm)
        end_time = time.time()
        cluster_pred = optics_model.labels_
        
        # Evaluate clustering
        metrics = EvaluationUtils.evaluate(X, class_labels, cluster_pred)
        current_algorithm = f'Optics, {dist}, {algorithm})'
        execution_time = end_time - start_time
        
        # Create reachability plot
        plt.subplot(len(distances), len(algorithms), plot_index)
        plt.plot(full_optics.reachability_[full_optics.ordering_])
        plt.title(f'Reachability Plot: {current_algorithm}')
        plt.xlabel('Points (ordered)')
        plt.ylabel('Reachability Distance')
        
        results.append({
            'Algorithm': current_algorithm,
            **metrics,
            'Time': execution_time
        })
        
        plot_index += 1

# Adjust layout and save plot
plt.tight_layout()
plot_path = os.path.join(dataset_path, 'Results/Plots/optics_reachability_plots.png')
plt.savefig(plot_path)

# Save results
results_df = pd.DataFrame(results)
csv_path = os.path.join(dataset_path, 'Results/CSVs/optics_results.csv')
results_df.to_csv(csv_path, index=False)
