# OPTICS results extraction
import os
import pandas as pd
import time
import numpy as np
from Classes.EvaluationUtils import EvaluationUtils
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt


if __name__ == "__main__" or "__mp_main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

# Load data
data_path = os.path.join(dataset_path, "Preprocessing", "hepatitis.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0', 'Class']).values

# Test configurations
distances = ['euclidean']#, 'manhattan', 'cosine']
algorithms = ['auto', 'brute', 'ball_tree', 'kd_tree']
colors = {'euclidean': 'b', 'manhattan': 'g', 'cosine': 'r'}  # Color mapping

# Perform clustering
results = []

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()  # To easily iterate through axes

for dist in distances:
    plot_index = 0
    current_algorithms = algorithms if dist != 'cosine' else ['auto', 'brute']
    for algorithm in current_algorithms:
        # Apply OPTICS
        start_time = time.time()
        optics = OPTICS(metric=dist, algorithm=algorithm, min_samples=15, xi=0.03,min_cluster_size=0.03)
        cluster_pred = optics.fit_predict(X)
        end_time = time.time()
        metrics = EvaluationUtils.evaluate(X, class_labels, cluster_pred)
        current_algorithm = f'(Optics, {dist}, {algorithm})'
        execution_time = end_time - start_time
        results.append({
            'Algorithm': current_algorithm,
            **metrics,
            'Time': execution_time
        })

        # Reachability distances and ordering
        reachability = optics.reachability_[optics.ordering_ < np.inf]
        ordering = optics.ordering_

        # Plot on the corresponding subplot
        ax = axes[plot_index]
        clusters = optics.labels_[optics.ordering_]
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            ax.vlines(x=np.where(mask)[0], ymin=0, ymax=reachability[mask], color=colors[dist], linewidth=1)


        # ax.plot(range(len(reachability)), reachability[ordering],'.' ,markersize=4, color=colors[dist], label=f'Distance: {dist.capitalize()}')
        # ax.vlines(x=range(len(reachability)), ymin=0, ymax=reachability[ordering], color=colors[dist], linewidth=2,
                  #label=f'Distance: {dist.capitalize()}')
        # Title and labels
        ax.set_title(f'OPTICS ({algorithm})', fontsize=10)
        ax.set_xlabel('Cluster Ordering', fontsize=9)
        ax.set_ylabel('Reachability Distance', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        #ax.legend(loc='upper right', fontsize=8)

        plot_index += 1
        if plot_index >= 4:  # Ensure we only fill 4 subplots
            break
        if plot_index >= 4:
            break

plt.tight_layout()

# Save the figure to the specified path
plot_filename = 'optics_reachability.png'
plot_path = os.path.join(dataset_path, 'Analysis/plots_and_tables/OPTICS', plot_filename)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()


# Save results
results_df = pd.DataFrame(results)
csv_path = os.path.join(dataset_path, 'Results/CSVs/optics_results.csv')
results_df.to_csv(csv_path, index=False)


