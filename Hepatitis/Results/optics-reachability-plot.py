import os
import pandas as pd
import numpy as np
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
data_path = os.path.join(dataset_path, "Preprocessing", "hepatitis.csv")
data = pd.read_csv(data_path)
class_labels = data['Class']
X = data.drop(columns=['Unnamed: 0', 'Class']).values

# Test configurations
distances = ['euclidean', 'manhattan', 'cosine']
algorithm = 'brute'

# Colors for each distance metric
colors = {
    'euclidean': '#1E90FF',  # Dodger Blue
    'manhattan': '#FF4500',  # Orange Red
    'cosine': '#32CD32'  # Lime Green
}

# Results to collect
results = []

# Create individual plots for each distance metric
for dist in distances:
    # Create figure
    plt.figure(figsize=(8, 6))

    # Clustering
    optics_model, full_optics = clustering(X, dist, algorithm)
    cluster_pred = optics_model.labels_

    # cgpt
    # Extract reachability distances
    reachability = full_optics.reachability_[full_optics.ordering_ < np.inf]
    ordering = full_optics.ordering_

    # Plot the reachability plot
    plt.plot(range(len(reachability)), reachability[ordering], 'g.', markersize=2)
    #plt.axhline(y=10, color='r', linestyle='--', label='Epsilon threshold')
    # cgpt

    # Evaluate clustering
    metrics = EvaluationUtils.evaluate(X, class_labels, cluster_pred)
    current_algorithm = f'Optics_{dist}_{algorithm}'

    # Filter out infinite values
    # reachability_values = full_optics.reachability_[full_optics.reachability_ < np.inf]

    # Create histogram
    #plt.hist(reachability,
    #         bins=50,
    #         color=colors[dist],
    #         alpha=0.7,
    #         density=True)

    plt.title(f'Reachability Distance Distribution: {dist.capitalize()} Distance', fontsize=14)
    plt.xlabel('Points', fontsize=12)
    plt.ylabel('Reachability Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save individual plot
    plot_filename = f'optics_reachability_histogram_{dist}.png'
    plot_path = os.path.join(dataset_path, 'Analysis/plots_and_tables/OPTICS', plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Collect results
    result_entry = {
        'Algorithm': current_algorithm,
        **metrics
    }
    results.append(result_entry)

